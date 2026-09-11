"""记忆召回 + 独立注入（票 24 / #31）。

按相似度取 top-k 相关事实，作为一块**独立的「已知信息」**注入；
记忆**不进 sources、不进检索候选池、不改变引用校验的结论**，且**按用户隔离**。

缝：stub 嵌入（相似度只有 1 / 0，阈值无歧义）+ 内存记忆存储 —— 确定性、无网络、无真实 LLM。
"""
from __future__ import annotations

from app.core.memory import InMemoryMemoryStore, MemoryRecall
from app.core.prompt import format_memory
from tests.helpers import StubEmbedding, register_and_kb, wait_until

FACT = "用户在跟进电池项目"
QUERY = "电池进展如何"


# ---------- 纯函数：召回 ----------

def test_recall_returns_only_relevant_and_capped():
    st = InMemoryMemoryStore()
    st.add("u1", ["用户在跟进电池项目", "用户偏好中文", "用户负责电池采购"])
    r = MemoryRecall(store=st, embedding=StubEmbedding(), top_k=1, min_score=0.9)
    assert [h["content"] for h in r.recall("u1", "电池进展如何")] == ["用户在跟进电池项目"]


def test_recall_caps_at_top_k():
    st = InMemoryMemoryStore()
    st.add("u1", ["用户在做电池项目{}".format(i) for i in range(6)])
    r = MemoryRecall(store=st, embedding=StubEmbedding(), top_k=2, min_score=0.9)
    assert len(r.recall("u1", "电池项目")) == 2


def test_recall_empty_store_returns_nothing():
    r = MemoryRecall(store=InMemoryMemoryStore(), embedding=StubEmbedding())
    assert r.recall("nobody", "随便问") == []


def test_recall_drops_below_threshold():
    """有记忆但都不相关 → 零召回（对应「无相关记忆时零注入」）。"""
    st = InMemoryMemoryStore()
    st.add("u1", ["用户偏好中文"])
    r = MemoryRecall(store=st, embedding=StubEmbedding(), min_score=0.9)
    assert r.recall("u1", "比亚迪营收") == []


def test_recall_is_per_user():
    """取不到他人的记忆 —— 用不同内容存两条，才分得清召回的是谁那条。"""
    st = InMemoryMemoryStore()
    st.add("u1", ["用户在跟进电池项目"])
    st.add("u2", ["用户负责电池采购"])
    r = MemoryRecall(store=st, embedding=StubEmbedding())
    assert [h["content"] for h in r.recall("u1", "电池进展如何")] == ["用户在跟进电池项目"]
    assert [h["content"] for h in r.recall("u2", "电池进展如何")] == ["用户负责电池采购"]
    assert r.recall("u3", "电池进展如何") == []


def test_long_fact_is_truncated_with_ellipsis():
    """超长事实截断要看得出被截了 —— 半句事实会被模型读成另一句。"""
    st = InMemoryMemoryStore()
    st.add("u1", ["电池" + "长" * 100])
    r = MemoryRecall(store=st, embedding=StubEmbedding(), max_chars=10)
    got = r.recall("u1", "电池")[0]["content"]
    assert len(got) == 10 and got.endswith("…")


def test_format_memory_is_none_when_empty():
    """无召回 → 不产生注入块。"""
    assert format_memory([]) is None
    assert format_memory(None) is None
    assert format_memory([{"content": "甲"}]) == "- 甲"


# ---------- 接进问答链路 ----------

def _seed(client, name, facts=None, with_doc=False):
    """注册用户 + 建库（可选上传文档并等入库）+ 装上带 stub 记忆缝的 Runtime。

    返回 (headers, kb_id, uid, store, rt)。文档必须在装上 rt **之后**上传，
    否则它进的是上一个运行时的（空）向量库，检索就命中不到了。
    """
    import app.api.deps as deps
    from app.core.container import build_runtime

    H, uid, kb_id = register_and_kb(client, name)

    store = InMemoryMemoryStore()
    if facts:
        store.add(uid, facts)
    rt = build_runtime()
    rt.memory_store = store
    deps._runtime = rt

    if with_doc:
        up = client.post("/api/v1/documents?kb_id={}".format(kb_id), headers=H,
                         files={"file": ("manual.txt", "比亚迪安全手册，关于电池和充电的规范说明。",
                                         "text/plain")}).json()
        assert wait_until(lambda: client.get("/api/v1/documents/{}".format(up["id"]), headers=H)
                          .json().get("status") in ("indexed", "failed")), "文档未入库"
        d = client.get("/api/v1/documents/{}".format(up["id"]), headers=H).json()
        assert d["status"] == "indexed", d

    rt.embedding = StubEmbedding()   # 只换召回用的嵌入缝，检索走 rt.retriever 自己的
    return H, kb_id, uid, store, rt


def _prepare(rt, uid, kb_id, question, session_id=None):
    from app.db.session import SessionLocal
    from app.models.entities import User
    from app.services import chat_service

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uid).first()
        return chat_service.prepare(db, rt, user, kb_id, question, session_id)
    finally:
        db.close()


def test_prepare_injects_memory_as_its_own_block(client):
    _H, kb_id, uid, _store, rt = _seed(client, "recall1", [FACT])
    prep = _prepare(rt, uid, kb_id, QUERY)

    assert prep.trace["memory_recalled"] == 1
    oi = prep.prompt.index("【已知信息】")
    assert prep.prompt.index(FACT) > oi            # 事实在【已知信息】块内
    assert prep.trace["memory_top_score"] == 1.0


def test_zero_injection_when_nothing_relevant(client):
    """有记忆但都不相关 → 不塞空块。"""
    _H, kb_id, uid, _store, rt = _seed(client, "recall2", ["用户偏好中文"])
    prep = _prepare(rt, uid, kb_id, "比亚迪2025年营收")

    assert "【已知信息】" not in prep.prompt
    assert prep.trace["memory_recalled"] == 0


def test_zero_injection_when_store_empty(client):
    _H, kb_id, uid, _store, rt = _seed(client, "recall3", [])
    prep = _prepare(rt, uid, kb_id, QUERY)
    assert "【已知信息】" not in prep.prompt


def test_memory_never_becomes_a_source_nor_a_candidate(client):
    """有文档、有召回：记忆在 prompt 里，但候选池与来源与「无记忆」时逐个一致。"""
    _H, kb_id, uid, _store, rt = _seed(client, "recall4", [FACT], with_doc=True)
    with_mem = _prepare(rt, uid, kb_id, QUERY)
    assert with_mem.trace["memory_recalled"] == 1
    assert with_mem.candidates, "本例应有检索候选，否则断言无意义"

    rt.memory_store = InMemoryMemoryStore()        # 换成空记忆
    without_mem = _prepare(rt, uid, kb_id, QUERY)

    assert [c["chunk_id"] for c in with_mem.candidates] == [c["chunk_id"] for c in without_mem.candidates]
    assert [s["chunk_id"] for s in with_mem.sources] == [s["chunk_id"] for s in without_mem.sources]
    assert all(FACT not in s["text"] for s in with_mem.sources)


def test_citation_verdict_is_unchanged_by_memory(client):
    """同一条答案，有/无记忆时引用校验的结论**完整**一致。"""
    _H, kb_id, uid, _store, rt = _seed(client, "recall5", [FACT], with_doc=True)
    with_mem = _prepare(rt, uid, kb_id, QUERY)

    rt.memory_store = InMemoryMemoryStore()
    without_mem = _prepare(rt, uid, kb_id, QUERY)

    assert with_mem._ccit.has_sources is True       # 本例有来源，才不会两边都 False 而恒真
    assert with_mem._ccit == without_mem._ccit      # 完整比对（含 coverage / stable_ids / notes）


def test_memory_is_counted_against_the_context_budget(client, monkeypatch):
    """记忆注入同样吃预算：注入后历史的可用额度相应减少，两块合计才不会撑爆。"""
    import app.config as cfg

    _H, kb_id, uid, _store, rt = _seed(client, "recall6", [FACT])
    monkeypatch.setattr(cfg.get_settings(), "context_token_budget", 500)

    with_mem = _prepare(rt, uid, kb_id, QUERY)
    rt.memory_store = InMemoryMemoryStore()
    without_mem = _prepare(rt, uid, kb_id, QUERY)

    memory_text = format_memory([{"content": FACT}])
    assert without_mem.trace["context_budget"] == 500
    assert with_mem.trace["context_budget"] == 500 - rt.token_counter.count(memory_text)


def test_off_switch_disables_recall(client, monkeypatch):
    """关闭记忆开关：即使有相关记忆也不注入（行为与今天一致）。"""
    import app.config as cfg

    _H, kb_id, uid, _store, rt = _seed(client, "recall7", [FACT])
    monkeypatch.setattr(cfg.get_settings(), "memory_enabled", False)
    prep = _prepare(rt, uid, kb_id, QUERY)

    assert "【已知信息】" not in prep.prompt
    assert prep.trace["memory_recalled"] == 0


def test_recall_failure_is_bypassed(client):
    """召回挂了也不该拖垮回答 —— 当作「无记忆」继续。"""
    _H, kb_id, uid, _store, rt = _seed(client, "recall8", [FACT])

    def boom(user_id, question):
        raise RuntimeError("memory store down")

    rt.recall_memory = boom
    prep = _prepare(rt, uid, kb_id, QUERY)
    assert "【已知信息】" not in prep.prompt
    assert prep.trace["memory_recalled"] == 0


# ---------- 跨会话（端到端：写入 → 召回） ----------

class RecordingLLM:
    """记录每次被喂的 prompt。is_fake=False 使链路走真实分支。

    问题改写做「恒等改写」（把原问题原样返回），保证后续检索 / 召回用的查询仍带关键词。
    """

    is_fake = False

    def __init__(self):
        self.prompts = []

    def stream(self, messages):
        content = messages[-1]["content"]
        self.prompts.append(content)
        if "查询改写助手" in content:
            yield content.rsplit("当前问题：", 1)[-1]
        else:
            yield "（回答）"


class _Factory:
    def __init__(self, llm):
        self.llm = llm

    def build(self, cfg):
        return self.llm


class StubExtractor:
    def __init__(self, facts):
        self.facts = facts

    def extract(self, question, answer):
        return list(self.facts)


def test_cross_session_recall_end_to_end(client):
    """会话 A 里用户告知的事实，在会话 B（新会话）被召回并进入 LLM 看到的 prompt。"""
    import app.api.deps as deps
    from app.core.byok import InMemoryUserLLMConfigStore, LLMConfig
    from app.core.container import build_runtime

    H, uid, kb = register_and_kb(client, "recallx")

    llm = RecordingLLM()
    store = InMemoryMemoryStore()
    rt = build_runtime()
    rt.memory_store = store
    rt.llm_factory = _Factory(llm)
    rt.fact_extractor_factory = lambda _llm: StubExtractor([FACT])
    cfg_store = InMemoryUserLLMConfigStore()
    cfg_store.set(uid, LLMConfig(base_url="https://x/v1", api_key="k", model="m"))
    rt.user_llm_config_store = cfg_store
    deps._runtime = rt
    rt.embedding = StubEmbedding()

    sess_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    sess_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    client.post("/api/v1/chat/stream", headers=H,
                json={"kb_id": kb, "question": "我在跟电池项目", "session_id": sess_a, "stream": True})
    assert wait_until(lambda: bool(store.list(uid))), "会话 A 的事实未落库"

    before = len(llm.prompts)
    client.post("/api/v1/chat/stream", headers=H,
                json={"kb_id": kb, "question": QUERY, "session_id": sess_b, "stream": True})

    assert any(FACT in p for p in llm.prompts[before:]), "会话 B 的 prompt 里应召回会话 A 的事实"
