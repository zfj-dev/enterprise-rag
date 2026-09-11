"""per-query 用量记账（票 27 / #34）：每次生成记一条用量，口径来源必须写清楚。

优先级：**provider usage（账单口径）> 本地分词器（估算口径）> 不可用**。
「不可用」不许记成 0 —— 那会让「这次花了 0」看起来像结论（spec 0003 的降级诚实性同源）。
"""
from __future__ import annotations

from app.core.llm import ToolCall
from app.core.usage import (SOURCE_LOCAL, SOURCE_PROVIDER, SOURCE_UNAVAILABLE, build_usage)
from tests.helpers import register_and_kb

PROMPT = "甲乙丙丁"
ANSWER = "戊己"


class LabeledCounter:
    """带口径的计数器（label 非空 = 真实分词器在）。每字 1 token，便于精确断言。"""

    label = "Qwen/test"

    def count(self, text: str) -> int:
        return len(text or "")


class NoLabelCounter:
    """没有真实分词器（label 为空）—— 只配用来做预算取舍，不许拿来对外报数。"""

    def count(self, text: str) -> int:
        return len(text or "")


# ---------- 口径优先级（纯函数） ----------

def test_provider_usage_wins_and_is_labelled_as_the_bill():
    rec = build_usage(prompt_text=PROMPT, answer_text=ANSWER, model="qwen-plus",
                      provider_usage={"prompt_tokens": 120, "completion_tokens": 30},
                      token_counter=LabeledCounter())

    assert (rec["input_tokens"], rec["output_tokens"]) == (120, 30)
    assert rec["source"] == SOURCE_PROVIDER
    assert rec["model"] == "qwen-plus"
    assert "账单" in rec["source_note"]        # 一眼看出这是账单口径


def test_without_provider_usage_it_falls_back_to_the_local_tokenizer():
    rec = build_usage(prompt_text=PROMPT, answer_text=ANSWER, model="qwen-plus",
                      provider_usage=None, token_counter=LabeledCounter())

    assert (rec["input_tokens"], rec["output_tokens"]) == (4, 2)
    assert rec["source"] == SOURCE_LOCAL
    assert "估算" in rec["source_note"] and "Qwen/test" in rec["source_note"]   # 口径写进记录


def test_with_neither_it_is_unavailable_and_never_zero():
    """两者皆无 -> 不可用，**不记 0**：0 是「没花 token」，与「量不到」是两回事。"""
    rec = build_usage(prompt_text=PROMPT, answer_text=ANSWER, model="qwen-plus",
                      provider_usage=None, token_counter=NoLabelCounter())

    assert rec["source"] == SOURCE_UNAVAILABLE
    assert rec["input_tokens"] is None and rec["output_tokens"] is None
    assert "不可用" in rec["source_note"]
    assert rec["input_tokens"] != 0 and rec["output_tokens"] != 0


def test_a_partial_provider_usage_is_not_padded_with_zero():
    """provider 只给了输入侧 —— 不能把输出侧补成 0（那是编出来的数字），退回本地口径。"""
    rec = build_usage(prompt_text=PROMPT, answer_text=ANSWER, model="m",
                      provider_usage={"prompt_tokens": 99}, token_counter=LabeledCounter())

    assert rec["source"] == SOURCE_LOCAL
    assert rec["output_tokens"] == 2          # 本地数的，不是补的 0


def test_the_record_is_a_plain_dict_so_it_can_be_stored():
    rec = build_usage(prompt_text="", answer_text="", model="m", token_counter=LabeledCounter())
    assert set(rec) == {"model", "input_tokens", "output_tokens", "source", "source_note"}


# ---------- provider 那一侧：usage 真的能拿到 ----------

def test_the_streaming_call_asks_for_usage_and_parses_the_final_chunk(monkeypatch):
    """usage 在**最后一个 chunk** 里、而且那一块 choices 为空 —— 解析正文失败不能顺手把它丢掉。

    这是流式协议最容易漏的一环：老代码 `json["choices"][0]` 会在 usage 块上抛 IndexError，
    被 except 吞掉，usage 就永远拿不到了。
    """
    import httpx

    from app.core.llm import CloudLLM

    lines = ['data: {"choices":[{"delta":{"content":"你"}}]}',
             'data: {"choices":[{"delta":{"content":"好"}}]}',
             'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2}}',
             "data: [DONE]"]
    sent: list = []

    class FakeResp:
        def raise_for_status(self): pass
        def iter_lines(self): return iter(lines)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, *a, **k):
            sent.append(k.get("json"))
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    llm = CloudLLM(base_url="https://x/v1", api_key="k", model="m")

    assert "".join(llm.stream([{"role": "user", "content": "hi"}])) == "你好"
    assert llm.last_usage == {"prompt_tokens": 11, "completion_tokens": 2}
    assert sent[0]["stream_options"] == {"include_usage": True}     # 不主动要，provider 不会给


def test_a_failed_call_clears_the_previous_usage(monkeypatch):
    """没配 Key 时不该把**上一次**的 usage 留在身上 —— 那会被记到这一次头上。"""
    from app.core.llm import CloudLLM

    llm = CloudLLM(base_url="https://x/v1", api_key="", model="m")
    llm.last_usage = {"prompt_tokens": 99, "completion_tokens": 99}

    assert "".join(llm.stream([{"role": "user", "content": "hi"}]))
    assert llm.last_usage is None


# ---------- 接进问答链路 ----------

class UsageLLM:
    """非假模型；流式跑完时给出 provider usage（模拟 OpenAI 兼容的最后一个 chunk）。"""

    is_fake = False

    def __init__(self, usage=None, model="qwen-plus"):
        self.model = model
        self._usage = usage
        self.last_usage = None

    def stream(self, messages):
        yield "答案"
        self.last_usage = self._usage


def _setup(client, name: str, *, llm=None, counter=None):
    """注册 + 建库，并给运行时装上可控的模型与计数器。"""
    import app.api.deps as deps
    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = build_runtime()
    rt.token_counter = counter if counter is not None else LabeledCounter()
    if llm is not None:
        rt.llm = llm
    deps._runtime = rt
    return H, uid, kb, user, db, rt


def test_a_query_records_one_row_saying_which_口径(client):
    from app.services import chat_service

    _, uid, kb, user, db, rt = _setup(client, "usage_row",
                                      llm=UsageLLM({"prompt_tokens": 120, "completion_tokens": 30}))
    try:
        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert len(rows) == 1
    assert rows[0]["source"] == SOURCE_PROVIDER
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (120, 30)
    assert out["trace"]["usage"]["source"] == SOURCE_PROVIDER     # 挂在 per-query trace 上


def test_the_record_is_per_user(client):
    from app.services import chat_service

    _, uid_a, kb_a, user_a, db, rt = _setup(client, "usage_iso_a")
    H_b, uid_b, kb_b = register_and_kb(client, "usage_iso_b")
    try:
        from app.models.entities import User

        user_b = db.query(User).filter(User.id == uid_b).first()
        chat_service.answer(db, rt, user_a, kb_a, "甲的问题")
        chat_service.answer(db, rt, user_b, kb_b, "乙的问题")

        assert len(rt.usage_store.list(uid_a)) == 1
        assert len(rt.usage_store.list(uid_b)) == 1
        assert rt.usage_store.list(uid_a)[0]["user_id"] == uid_a
    finally:
        db.close()


def test_the_switch_off_records_nothing(client, monkeypatch):
    """关掉成本功能：不记账（行为与今天一致）。"""
    from app.services import chat_service
    from app.config import get_settings

    _, uid, kb, user, db, rt = _setup(client, "usage_off")
    try:
        monkeypatch.setattr(get_settings(), "cost_enabled", False)
        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert rows == []


def test_the_done_event_carries_the_usage(client):
    """per-query trace 之外，流式 done 事件也带上 —— 前端/评测都能看到这笔。"""
    from tests.helpers import sse_events

    H, uid, kb, user, db, rt = _setup(client, "usage_sse",
                                      llm=UsageLLM({"prompt_tokens": 7, "completion_tokens": 3}))
    db.close()
    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "问", "stream": True})
    done = sse_events(r.text)[-1]

    assert done["type"] == "done"
    assert done["usage"]["source"] == SOURCE_PROVIDER
    assert done["usage"]["input_tokens"] == 7


# ---------- 代理链路：多步调用要合计，不能只记最后一次 ----------

def test_the_agent_sums_the_provider_usage_over_steps():
    from app.core.agent import run_agent
    from app.core.tools import Tool, ToolRegistry
    from app.mcp.client import InProcessTransport

    class StepLLM:
        is_fake = False

        def __init__(self):
            self.n = 0
            self.last_usage = None

        def stream(self, messages):
            yield ""

        def chat_with_tools(self, messages, tools=None):
            self.n += 1
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
            if tools:
                return {"content": "", "tool_calls": [ToolCall(id="c1", name="Calculator",
                                                               arguments={"expression": "1+1"})]}
            return {"content": "等于 2", "tool_calls": []}

    tool = Tool(name="Calculator", description="算",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: {"value": 2})
    got = run_agent("1+1", llm=StepLLM(), transport=InProcessTransport(ToolRegistry(tools=[tool])),
                    max_steps=2)

    assert got["trace"]["llm_usage"]["prompt_tokens"] == 30      # 三轮各 10
    assert got["trace"]["llm_usage"]["completion_tokens"] == 15


# ---------- 审查抓到的那两类「记到别人头上」 ----------

class TwoPhaseLLM:
    """按 prompt 分辨调用类型：改写 / **生成**（账单 100/20）/ 之后的那些（999）。

    记账必须记生成那一笔 —— 快照取晚了，引用校验那次就会把它盖掉。
    """

    is_fake = False

    def __init__(self):
        self.model = "qwen-plus"
        self.last_usage = None

    def stream(self, messages):
        content = messages[-1]["content"]
        if "查询改写助手" in content:                      # 问题改写（prepare 里先跑）
            self.last_usage = {"prompt_tokens": 5, "completion_tokens": 2}
            yield content.rsplit("当前问题：", 1)[-1].strip()
        elif "【用户问题】" in content:                    # 真正的生成
            self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}
            yield "答案"
        else:                                              # 引用校验 / 事实抽取等
            self.last_usage = {"prompt_tokens": 999, "completion_tokens": 999}
            yield "{}"


def test_the_generation_usage_is_snapshotted_before_the_later_calls(client):
    """引用校验 / 事实抽取会接着调同一个模型实例 —— 记的必须是**生成**那一笔。"""
    from app.services import chat_service

    _, uid, kb, user, db, rt = _setup(client, "usage_snapshot", llm=TwoPhaseLLM())
    try:
        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert len(rows) == 1
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (100, 20)


def test_a_cache_hit_records_nothing(client):
    """缓存命中没有生成调用 —— 拿没发出去的 prompt 估一个数，等于给没发生的事记账。"""
    from app.services import chat_service

    _, uid, kb, user, db, rt = _setup(client, "usage_cache")
    try:
        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        assert len(rt.usage_store.list(uid)) == 1        # 第一次是真生成，记一笔

        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert len(rows) == 1                                 # 命中缓存那次不记


def test_a_provider_that_rejects_stream_options_still_answers(monkeypatch):
    """provider 不认 stream_options 时去掉它重试一次 —— 不能因为多要一个字段就把整条流打断。"""
    import httpx

    from app.core.llm import CloudLLM

    attempts: list = []

    class FakeResp:
        def __init__(self, status): self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad", request=httpx.Request("POST", "https://x/v1"),
                                            response=self)

        def iter_lines(self):
            return iter(['data: {"choices":[{"delta":{"content":"好"}}]}', "data: [DONE]"])

        def __enter__(self): return self
        def __exit__(self, *a): return False

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def stream(self, *a, **k):
            has = "stream_options" in (k.get("json") or {})
            attempts.append(has)
            return FakeResp(400 if has else 200)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    llm = CloudLLM(base_url="https://x/v1", api_key="k", model="m")

    assert "".join(llm.stream([{"role": "user", "content": "hi"}])) == "好"
    assert attempts == [True, False]          # 先带、被拒、去掉再来
    assert llm.last_usage is None             # 这次拿不到账单口径 —— 记账会回退本地
