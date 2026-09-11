"""代理循环（票 11 / #18）：脚本化 LLM + 进程内传输 —— 确定性、无网络、不起真 MCP 进程。"""
from __future__ import annotations

from app.core.agent import run_agent
from app.core.tools import Tool, ToolRegistry
from app.core.llm import FakeLLM, ToolCall
from app.mcp.client import InProcessTransport


class ScriptedLLM:
    """按脚本吐回复（脚本用完就吐终答）。记录每次调用，便于断言「结果有没有回灌」。"""

    is_fake = False

    def __init__(self, replies, fallback="（最后收敛作答）"):
        self.replies = list(replies)
        self.fallback = fallback
        self.calls = []

    def stream(self, messages):
        yield self.fallback

    def chat_with_tools(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not tools:
            # 与真实现的降级路径一致：不给工具就是普通一问一答（收敛那一步走这里）
            return {"content": self.fallback, "tool_calls": []}
        if self.replies:
            return self.replies.pop(0)
        return {"content": self.fallback, "tool_calls": []}


def _calc_call(expression, cid="c1"):
    return ToolCall(id=cid, name="Calculator", arguments={"expression": expression})


def _calc_transport():
    from app.core.tools import default_registry
    return InProcessTransport(default_registry())


# ---------- 正常一步 ----------

def test_one_tool_call_then_an_answer():
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("(12.5-10)/10*100")]},
        {"content": "同比增长 25%", "tool_calls": []},
    ])
    got = run_agent("算出同比增长率", llm=llm, transport=_calc_transport())

    assert got["answer"] == "同比增长 25%"
    assert got["stopped"] == "answered"
    assert len(got["steps"]) == 1
    step = got["steps"][0]
    assert step["tool"] == "Calculator"
    assert step["arguments"] == {"expression": "(12.5-10)/10*100"}
    assert "25.0" in step["summary"]
    assert step["ok"] is True and isinstance(step["ms"], float)
    assert set(step) == {"tool", "arguments", "summary", "ms", "ok"}


def test_the_tool_result_is_fed_back_to_the_model():
    """结果要回灌，模型才可能据此作答 —— 不然第二步等于没拿到数据。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("1+1")]},
        {"content": "等于 2", "tool_calls": []},
    ])
    run_agent("1+1 等于几", llm=llm, transport=_calc_transport())

    second = llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "c1"
    assert '"value": 2' in second[-1]["content"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "Calculator"


# ---------- 三条终止路径 ----------

def test_max_steps_converges_instead_of_raising():
    """一直在调工具也不能无限跑：到上限就**收敛作答**，绝不抛穿。"""
    llm = ScriptedLLM([{"content": "", "tool_calls": [_calc_call("1+1")]}] * 10)
    got = run_agent("没完没了的问题", llm=llm, transport=_calc_transport(), max_steps=3)

    assert got["stopped"] == "max_steps"
    assert got["answer"] == "（最后收敛作答）"
    assert len(got["steps"]) == 3                       # 正好跑满上限，没多跑
    assert llm.calls[-1]["tools"] is None               # 最后那次是**不带工具**逼它作答


def test_tool_error_is_recorded_and_the_run_still_converges():
    """单次工具失败不能把整轮丢掉 —— 记下来、继续跑；但报错的工具什么也没给，最终拒答。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("data[0]")]},     # 一定会被拒
        {"content": "换个说法：这个表达式不合法", "tool_calls": []},
    ])
    got = run_agent("算个非法表达式", llm=llm, transport=_calc_transport())

    assert got["steps"][0]["ok"] is False
    assert "只允许数字与算术运算" in got["steps"][0]["summary"]
    assert got["stopped"] == "answered"
    assert got["trace"]["self_check"] == "refused"
    assert "无法确定" in got["answer"]


def test_transport_blowing_up_is_recorded_not_raised():
    class Boom:
        """工具清单正常，但一调就炸 —— 模拟传输断掉。"""

        def __init__(self, registry):
            self._registry = registry

        def list_tools(self):
            return self._registry.specs()

        def call_tool(self, name, arguments=None):
            raise RuntimeError("传输断了")

    llm = ScriptedLLM([
        {"content": "", "tool_calls": [ToolCall(id="c1", name="Calculator",
                                                arguments={"expression": "1+1"})]},
        {"content": "工具没通，我按现有信息回答", "tool_calls": []},
    ])
    from app.core.tools import default_registry
    got = run_agent("问", llm=llm, transport=Boom(default_registry()))

    assert got["steps"][0]["ok"] is False
    assert "传输断了" in got["steps"][0]["summary"]
    assert got["stopped"] == "answered"


# ---------- 退化路径与来源 ----------

def test_fake_llm_degrades_to_a_single_step_answer():
    """演示模式的 Fake 不产生工具调用 —— 代理等价于一次普通生成，不崩。"""
    got = run_agent("随便问", llm=FakeLLM(), transport=_calc_transport())

    assert got["steps"] == []
    assert got["answer"]
    assert got["stopped"] == "answered"


def test_sources_from_a_tool_result_are_collected():
    """工具带回来的来源要收上来 —— 代理的答案照样得可追溯（票 14 会用）。"""
    tool = Tool(name="KbRetrieve", description="检索",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: {"sources": [{"chunk_id": "ch1", "page": 2}]})
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [ToolCall(id="k1", name="KbRetrieve", arguments={})]},
        {"content": "按资料，答案是 X", "tool_calls": []},
    ])
    got = run_agent("查一下", llm=llm, transport=InProcessTransport(ToolRegistry(tools=[tool])))

    assert got["sources"] == [{"chunk_id": "ch1", "page": 2}]


# ---------- 审查补的回归 ----------

def test_llm_blowing_up_mid_loop_does_not_raise():
    """spec：「任一路径…不得抛穿」—— 模型这一步出错也得收敛返回，不能把整轮丢掉。"""
    class Boom:
        is_fake = False

        def stream(self, messages):
            yield ""

        def chat_with_tools(self, messages, tools=None):
            raise RuntimeError("模型挂了")

    got = run_agent("问", llm=Boom(), transport=_calc_transport())

    assert got["stopped"] == "llm_error"                 # 如实说清是哪条路径终止的
    assert got["steps"] == []
    assert isinstance(got["answer"], str)                 # 空串也算返回，反正没抛


def test_llm_blowing_up_only_at_the_convergence_step_still_returns():
    class Flaky:
        is_fake = False

        def __init__(self):
            self.n = 0

        def stream(self, messages):
            yield ""

        def chat_with_tools(self, messages, tools=None):
            self.n += 1
            if tools:                                     # 循环内照常给工具调用
                return {"content": "", "tool_calls": [_calc_call("1+1")]}
            raise RuntimeError("收敛那步也挂了")

    got = run_agent("问", llm=Flaky(), transport=_calc_transport(), max_steps=1)

    assert got["stopped"] == "max_steps"
    assert len(got["steps"]) == 1


def test_every_run_reports_latency():
    """返回契约 {answer, sources, steps, latency, trace} —— 多步代价与自检结论都要看得见。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("1+1")]},
        {"content": "等于 2", "tool_calls": []},
    ])
    got = run_agent("1+1", llm=llm, transport=_calc_transport())

    assert set(got) == {"answer", "sources", "steps", "latency", "stopped", "trace"}
    assert got["latency"]["total_ms"] > 0
    assert got["latency"]["steps_ms"] == [got["steps"][0]["ms"]]
    assert set(got["trace"]) == {"self_check", "citation_coverage"}


def test_tool_failure_reason_reaches_the_model():
    """工具报了错，模型也要拿到原因 —— 否则它没法换个办法。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("data[0]")]},
        {"content": "换个数算", "tool_calls": []},
    ])
    run_agent("算个非法的", llm=llm, transport=_calc_transport())

    fed_back = llm.calls[1]["messages"][-1]
    assert fed_back["role"] == "tool"
    assert "error" in fed_back["content"] and "只允许数字与算术运算" in fed_back["content"]


# ---------- 票 14：引用与 1 步自检 ----------

_SRC = [{"chunk_id": "ch1", "doc_id": "d1", "doc_name": "年报.pdf", "page": 3,
         "text": "比亚迪2025年营业收入为803.96亿元", "score": 0.9}]


def _kb_tool(sources):
    """总是返回给定 sources 的检索工具（结果形状与 KbRetrieve 一致）。"""
    def handler(args):
        return {"sources": sources}
    return Tool(name="KbRetrieve", description="检索",
                input_schema={"type": "object", "properties": {}}, handler=handler)


def _kb_transport(sources):
    return InProcessTransport(ToolRegistry(tools=[_kb_tool(sources)]))


def _kb_then_answer(answer, *ids):
    return ([{"content": "", "tool_calls": [ToolCall(id=i, name="KbRetrieve", arguments={})]}
             for i in ids] + [{"content": answer, "tool_calls": []}])


class VerdictLLM(ScriptedLLM):
    """stream 返回逐句校验的 JSON —— 让 verify_claims 走真实分支（不是降级）。"""

    def stream(self, messages):
        yield '{"claims":[{"claim":"比亚迪2025年营业收入为803.96亿元。","supported":true}]}'


def test_refuses_when_retrieval_comes_back_empty():
    """检索跑了却一无所获 —— 沿用 no source → no claim，不许硬答。"""
    llm = ScriptedLLM(_kb_then_answer("营收大概八百亿上下", "k1"))
    got = run_agent("营收多少", llm=llm, transport=_kb_transport([]))

    assert got["sources"] == []
    assert "无法确定" in got["answer"]
    assert got["trace"]["self_check"] == "refused"


def test_refuses_when_no_tool_was_used_at_all():
    """一次工具都没调、直接凭记忆作答 —— 同样无依据可追溯（多步不是免检理由）。"""
    llm = ScriptedLLM([{"content": "比亚迪2025年营收803.96亿元", "tool_calls": []}])
    got = run_agent("营收多少", llm=llm, transport=_kb_transport(_SRC))

    assert "无法确定" in got["answer"]
    assert got["trace"]["self_check"] == "refused"


def test_arithmetic_answer_is_not_refused():
    """算数题的依据是 Calculator 的结果，不是文档 —— 不该被 no-claim 顶掉。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("1+1")]},
        {"content": "等于 2", "tool_calls": []},
    ])
    got = run_agent("1+1 等于几", llm=llm, transport=_calc_transport())

    assert got["answer"] == "等于 2"
    assert got["trace"]["self_check"] == "passed_with_tool"


def test_cited_answer_keeps_its_sources_with_doc_name_and_page():
    """"带引用"是可追溯的：来源里有文档名 + 页码，答案原文保留。"""
    llm = ScriptedLLM(_kb_then_answer("比亚迪2025年营业收入为803.96亿元。", "k1"))
    got = run_agent("营收多少", llm=llm, transport=_kb_transport(_SRC))

    assert got["answer"] == "比亚迪2025年营业收入为803.96亿元。"
    assert [(s["doc_name"], s["page"]) for s in got["sources"]] == [("年报.pdf", 3)]
    assert got["trace"]["self_check"] == "passed_with_citation"


def test_repeated_retrieval_does_not_duplicate_sources():
    """同一块被检索两次，来源里只出现一次 —— 引用清单不该重复。"""
    llm = ScriptedLLM(_kb_then_answer("按资料，803.96 亿元。", "k1", "k2"))
    got = run_agent("营收多少", llm=llm, transport=_kb_transport(_SRC))

    assert [s["chunk_id"] for s in got["sources"]] == ["ch1"]


def test_citation_coverage_lands_in_trace():
    llm = VerdictLLM(_kb_then_answer("比亚迪2025年营业收入为803.96亿元。", "k1"))
    got = run_agent("营收多少", llm=llm, transport=_kb_transport(_SRC))

    assert got["trace"]["citation_coverage"] == 1.0


def test_a_failed_tool_does_not_count_as_grounding():
    """工具报错等于没拿到东西 —— 不能因此就解锁凭记忆作答（审查抓到的漏拒答）。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("data[0]")]},          # 必被拒
        {"content": "我凭记忆答：营收803.96亿元", "tool_calls": []},
    ])
    got = run_agent("营收多少", llm=llm, transport=_calc_transport())

    assert got["steps"][0]["ok"] is False
    assert got["trace"]["self_check"] == "refused"
    assert "无法确定" in got["answer"]


def test_a_successful_side_tool_cannot_unlock_a_groundless_answer():
    """检索跑了却一无所获 —— 中间夹一次成功的 Calculator 也不解锁（模型能自己造这一步）。"""
    from app.core.tools import default_registry
    tools = default_registry().tools + [_kb_tool([])]
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [ToolCall(id="k1", name="KbRetrieve", arguments={})]},
        {"content": "", "tool_calls": [_calc_call("0+0", "c1")]},
        {"content": "比亚迪2025年营收803.96亿元", "tool_calls": []},
    ])
    got = run_agent("营收多少", llm=llm, transport=InProcessTransport(ToolRegistry(tools=tools)))

    assert [s["tool"] for s in got["steps"]] == ["KbRetrieve", "Calculator"]
    assert all(s["ok"] for s in got["steps"])          # 两个工具都"成功"
    assert got["trace"]["self_check"] == "refused"
    assert "无法确定" in got["answer"]


def test_fake_mode_skips_the_self_check():
    """演示模式没有工具调用，等价一次普通生成 —— 不该被拒答文案顶掉。"""
    got = run_agent("随便问", llm=FakeLLM(), transport=_kb_transport(_SRC))

    assert got["trace"]["self_check"] == "skipped"
    assert "无法确定" not in got["answer"]
