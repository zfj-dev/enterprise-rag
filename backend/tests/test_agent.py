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


def test_tool_error_degrades_instead_of_dropping_the_answer():
    """单次工具失败不能把整个回答丢掉 —— 记下来、继续，最后仍要有答案。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("data[0]")]},     # 一定会被拒
        {"content": "换个说法：这个表达式不合法", "tool_calls": []},
    ])
    got = run_agent("算个非法表达式", llm=llm, transport=_calc_transport())

    assert got["steps"][0]["ok"] is False
    assert "只允许数字与算术运算" in got["steps"][0]["summary"]
    assert got["answer"] == "换个说法：这个表达式不合法"
    assert got["stopped"] == "answered"


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
    """spec 的返回契约是 {answer, sources, steps, latency} —— 多步代价要看得见。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call("1+1")]},
        {"content": "等于 2", "tool_calls": []},
    ])
    got = run_agent("1+1", llm=llm, transport=_calc_transport())

    assert set(got) == {"answer", "sources", "steps", "latency", "stopped"}
    assert got["latency"]["total_ms"] > 0
    assert got["latency"]["steps_ms"] == [got["steps"][0]["ms"]]


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
