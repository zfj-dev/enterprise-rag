"""跨会话记忆：写入时异步抽取并按用户落库（票 23）。

抽取器与存储都注入 stub —— 确定性、无网络。
"""
from __future__ import annotations

import threading
import time

from app.core.memory import (DbMemoryStore, FactExtractor, InMemoryMemoryStore,
                             LlmFactExtractor)
from tests.helpers import offline_resolver as _resolver,  register_and_kb, wait_until


class StubLLM:
    """脚本化假 LLM。is_fake=False，使链路走"真实模型"分支。"""

    is_fake = False

    def __init__(self, out: str = "这是自带模型产出的答案。"):
        self.out = out
        self.prompts: list = []

    def stream(self, messages):
        self.prompts.append(messages)
        yield self.out


class StubExtractor(FactExtractor):
    def __init__(self, facts=None, boom: bool = False):
        self.facts = ["用户所在项目是 XX"] if facts is None else facts
        self.boom = boom
        self.seen: list = []

    def extract(self, question, answer):
        self.seen.append((question, answer))
        if self.boom:
            raise RuntimeError("extractor down")
        return list(self.facts)


class BlockingExtractor(FactExtractor):
    """抽取器卡在闸门上，用来证明回答不等它。"""

    def __init__(self, gate: threading.Event):
        self.gate = gate
        self.started = threading.Event()

    def extract(self, question, answer):
        self.started.set()
        self.gate.wait(10)
        return ["用户在等一个慢抽取器"]


# ---------- 抽取器 ----------

def test_llm_extractor_parses_dedups_and_caps():
    llm = StubLLM("- 用户在上海工作\n2. 用户偏好中文\n- 用户在上海工作\n\n")
    assert LlmFactExtractor(llm, max_facts=5).extract("q", "a") == ["用户在上海工作", "用户偏好中文"]


def test_llm_extractor_caps_at_max():
    llm = StubLLM("\n".join(f"事实{i}" for i in range(10)))
    assert len(LlmFactExtractor(llm, max_facts=3).extract("q", "a")) == 3


def test_extractor_separates_user_turn_from_answer_and_forbids_taking_either_as_fact():
    """只抽用户告知的事实：两段输入须**分开标注**，且明令不得把文档/助手回答当事实。

    抽取器的输入只有 (question, answer)，压根拿不到检索上下文 —— 文档内容唯一可能的入口
    是助手回答里引用的片段，所以这条"排除纪律"就是该边界上的唯一防线，只能断言到提示词。
    """
    llm = StubLLM("")
    LlmFactExtractor(llm).extract("我在做 XX 项目", "根据资料，XX 项目很好。")
    p = llm.prompts[0][0]["content"]

    user_at, answer_at = p.index("【用户提问】"), p.index("【助手回答】")
    assert user_at < answer_at                                # 两段分开、顺序固定
    assert user_at < p.index("我在做 XX 项目") < answer_at      # 提问落在用户段
    assert p.index("根据资料，XX 项目很好。") > answer_at       # 回答落在助手段
    # 排除纪律：不摘文档、不把助手自己的回答当事实
    assert "不要" in p and "文档" in p and "参考资料" in p and "助手" in p


# ---------- 存储 ----------

def test_inmemory_store_is_per_user():
    st = InMemoryMemoryStore()
    st.add("u1", ["事实A"])
    st.add("u2", ["事实B"])
    assert [f["content"] for f in st.list("u1")] == ["事实A"]
    assert [f["content"] for f in st.list("u2")] == ["事实B"]
    assert st.list("u3") == []


def test_db_store_roundtrip_and_isolation(client):
    st = DbMemoryStore()
    assert st.add("uA", ["事实A"], session_id="s1") == 1
    st.add("uB", ["事实B"])
    assert [f["content"] for f in st.list("uA")] == ["事实A"]
    assert [f["content"] for f in st.list("uB")] == ["事实B"]
    assert st.list("uC") == []
    assert st.add("uA", []) == 0


def test_db_store_list_follows_insertion_order(client):
    """同秒内插入的多条记忆，list 必须按插入顺序返回。

    回归：created_at 曾是 sqlite 的 CURRENT_TIMESTAMP（秒级），同秒写入的多行全部相等
    -> 排序退化到 uuid 主键（随机）-> 列表顺序任意。这里一批 8 条 + 跨批 5 条，都落在同一秒。
    """
    st = DbMemoryStore()
    batch = [f"事实{i}" for i in range(8)]
    st.add("uA", batch)
    assert [f["content"] for f in st.list("uA")] == batch

    st.add("uB", ["B1", "B2", "B3"])            # 两批之间也只隔几微秒
    st.add("uB", ["B4", "B5"])
    assert [f["content"] for f in st.list("uB")] == ["B1", "B2", "B3", "B4", "B5"]


# ---------- 接进问答链路 ----------

class _StubFactory:
    def build(self, cfg):
        return StubLLM()


def _setup(client, name: str, extractor):
    """注册用户 + 建库 + 装上"带记忆 stub"的 Runtime，返回 (headers, kb_id, uid, store)。"""
    import app.api.deps as deps
    from app.core.byok import InMemoryUserLLMConfigStore, LLMConfig
    from app.core.container import build_runtime

    H, uid, kb = register_and_kb(client, name)

    store = InMemoryMemoryStore()
    rt = build_runtime()
    rt.fact_extractor_factory = lambda llm: extractor
    rt.memory_store = store
    rt.llm_factory = _StubFactory()
    cfg_store = InMemoryUserLLMConfigStore()
    cfg_store.set(uid, LLMConfig(base_url="https://x/v1", api_key="k", model="m"))
    rt.user_llm_config_store = cfg_store
    rt.url_resolver = _resolver()   # 票 33：用之前会复查 base_url，测试里不查真 DNS
    deps._runtime = rt
    return H, kb, uid, store


def test_db_store_delete_is_per_user(client):
    """落库实现的删除也按用户下推：别人的、不存在的都删不掉。"""
    st = DbMemoryStore()
    st.add("uA", ["事实A", "事实B"])
    st.add("uB", ["别人的事实"])
    # 按内容取 id，不依赖 list 的顺序（本用例只验过滤；顺序由
    # test_db_store_list_follows_insertion_order 单独守）
    ids = {f["content"]: f["id"] for f in st.list("uA")}

    assert st.delete("uB", ids["事实A"]) is False   # 别人的删不掉
    assert st.delete("uA", "nope") is False         # 不存在的删不掉
    assert st.delete("uA", ids["事实A"]) is True
    assert [f["content"] for f in st.list("uA")] == ["事实B"]
    assert [f["content"] for f in st.list("uB")] == ["别人的事实"]


def test_chat_extracts_facts_per_user(client):
    """一轮问答后，事实被抽出并归属到该用户。"""
    ex = StubExtractor(["用户所在项目是 XX"])
    H, kb, uid, store = _setup(client, "mem1", ex)

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "我在做 XX 项目", "stream": True})
    assert r.status_code == 200

    assert wait_until(lambda: bool(store.list(uid))), "事实未落库"
    assert [f["content"] for f in store.list(uid)] == ["用户所在项目是 XX"]
    assert ex.seen and ex.seen[0][0] == "我在做 XX 项目"


def test_extraction_failure_does_not_break_answer(client):
    """抽取器挂了也必须旁路：回答照常返回。"""
    ex = StubExtractor(boom=True)
    H, kb, uid, store = _setup(client, "mem2", ex)

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "我在做 XX 项目", "stream": True})
    assert r.status_code == 200
    assert '"done"' in r.text or "done" in r.text
    assert wait_until(lambda: bool(ex.seen)), "抽取器应被调用过"
    assert store.list(uid) == []      # 失败 → 什么都没写，但也没崩


def test_memory_disabled_stores_nothing(client, monkeypatch):
    """总开关关闭时不抽取、不落库。"""
    import app.config as cfg

    ex = StubExtractor(["不该被写入"])
    H, kb, uid, store = _setup(client, "mem3", ex)
    monkeypatch.setattr(cfg.get_settings(), "memory_enabled", False)

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "我在做 XX 项目", "stream": True})
    assert r.status_code == 200
    time.sleep(0.3)
    assert store.list(uid) == []
    assert ex.seen == []


def test_slow_extraction_does_not_block_the_answer(client):
    """抽取变慢/挂住也不阻塞回答：回答先完整返回，事实稍后才落库。"""
    gate = threading.Event()
    ex = BlockingExtractor(gate)
    H, kb, uid, store = _setup(client, "mem4", ex)

    t0 = time.time()
    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "我在做 XX 项目", "stream": True})
    elapsed = time.time() - t0

    assert r.status_code == 200
    assert "done" in r.text                    # 回答与 done 事件已完整下发
    assert ex.started.wait(2), "抽取器应已在后台启动"
    assert elapsed < 2, "回答不该等待抽取（实际耗时 {:.1f}s）".format(elapsed)
    assert store.list(uid) == []               # 抽取器还卡着 → 尚未落库

    gate.set()                                 # 放行
    assert wait_until(lambda: bool(store.list(uid))), "放行后事实应落库"


def test_list_marker_stripping_keeps_digit_leading_facts():
    """去掉列表标记，但不吞掉以数字开头的正文（回归：lstrip 曾把「2024年营收」削成「年营收」）。"""
    llm = StubLLM(chr(10).join(
        ["- 用户在上海工作", "2. 2024年营收口径按合并报表", "1) 用户偏好中文"]))
    assert LlmFactExtractor(llm).extract("q", "a") == [
        "用户在上海工作", "2024年营收口径按合并报表", "用户偏好中文"]
