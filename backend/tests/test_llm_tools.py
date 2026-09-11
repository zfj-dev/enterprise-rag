"""LLM 带工具对话能力（票 09 / #16）：默认降级、provider 拒绝工具时不炸、原文流接口不变。"""
from __future__ import annotations

import types

from app.core.llm import CloudLLM, FakeLLM, ToolCall, parse_tool_calls


# ---------- 默认实现：明确降级 ----------

def test_fake_llm_degrades_to_a_single_step_answer():
    """不支持工具的模型不该抛穿 —— 当成普通一问一答，工具调用为空。"""
    got = FakeLLM().chat_with_tools([{"role": "user", "content": "问"}],
                                    tools=[{"type": "function"}])
    assert got["tool_calls"] == []
    assert got["content"]                      # FakeLLM 的固定回答照常给


def test_existing_stream_interface_is_untouched():
    """既有文本流接口的签名与行为一个字没动 —— 现有问答链路全靠它。"""
    assert "".join(FakeLLM().stream([{"role": "user", "content": "x"}]))
    assert FakeLLM().is_fake is True


# ---------- 工具调用解析 ----------

def test_parse_tool_calls_reads_name_and_arguments():
    raw = [{"id": "c1", "function": {"name": "calc", "arguments": '{"expression": "1+2"}'}}]
    assert parse_tool_calls(raw) == [ToolCall(id="c1", name="calc",
                                              arguments={"expression": "1+2"},
                                              raw='{"expression": "1+2"}')]


def test_parse_tool_calls_survives_broken_json():
    """模型偶尔吐不合法 JSON —— 解析不了就留空 dict + 原文，不炸。"""
    broken = parse_tool_calls([{"id": "c", "function": {"name": "calc", "arguments": "{坏"}}])
    assert broken[0].arguments == {} and broken[0].raw == "{坏"


def test_parse_tool_calls_skips_non_dict_items_instead_of_raising():
    """协议外的杂项不该把整轮对话带崩 —— docstring 说「这里不抛」就得真不抛。"""
    got = parse_tool_calls(["x", None, 42, {"function": {"name": "calc"}}])
    assert [c.name for c in got] == ["calc"]


def test_parse_tool_calls_wraps_non_object_json_per_the_documented_convention():
    """模型直接给了个数字：按 docstring 的约定塞进 {"value": ...}。"""
    assert parse_tool_calls([{"function": {"name": "calc", "arguments": "42"}}])[0].arguments         == {"value": 42}
def test_parse_tool_calls_tolerates_missing_pieces():
    assert parse_tool_calls(None) == []
    assert parse_tool_calls([]) == []
    assert parse_tool_calls([{"function": {"name": "x"}}])[0].arguments == {}
    assert parse_tool_calls([{}])[0].name == ""


# ---------- 真 provider ----------

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Client:
    def __init__(self, post):
        self._post = post

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        return self._post(url, json)


def _cloud(post):
    llm = CloudLLM(base_url="https://x/v1", api_key="k", model="m")
    llm._httpx = types.SimpleNamespace(Client=lambda **kw: _Client(post))
    return llm


def test_cloud_parses_a_returned_tool_call():
    payload = {"choices": [{"message": {"content": "我算一下", "tool_calls": [
        {"id": "c1", "function": {"name": "calc", "arguments": '{"expression": "2*3"}'}}]}}]}
    llm = _cloud(lambda url, body: _Resp(payload))

    got = llm.chat_with_tools([{"role": "user", "content": "算 2*3"}], tools=[{"type": "function"}])

    assert [c.name for c in got["tool_calls"]] == ["calc"]
    assert got["tool_calls"][0].arguments == {"expression": "2*3"}
    assert got["content"] == "我算一下"          # 内容也要一起带回来


def test_cloud_degrades_instead_of_raising_when_tools_are_refused():
    """provider 不吃 tools 参数时会炸 —— 必须降级成一问一答，不能抛穿。"""
    def boom(url, body):
        raise RuntimeError("tools unsupported")

    llm = _cloud(boom)
    llm.stream = lambda messages: iter(["降级", "回答"])       # 降级后走的就是这条

    got = llm.chat_with_tools([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert got == {"content": "降级回答", "tool_calls": []}


def test_cloud_without_tools_never_touches_the_wire():
    def boom(*a, **k):
        raise AssertionError("没给工具就不该发带工具的请求")

    llm = _cloud(boom)
    llm.stream = lambda messages: iter(["答"])
    assert llm.chat_with_tools([{"role": "user", "content": "x"}])["tool_calls"] == []


def test_parse_tool_calls_recovers_json_wrapped_in_a_code_block():
    """模型爱把 JSON 裹进 ``` 代码块 —— 抠出来，别当坏 JSON 丢掉。"""
    wrapped = "```json" + chr(10) + '{"expression": "1+1"}' + chr(10) + "```"
    got = parse_tool_calls([{"function": {"name": "calc", "arguments": wrapped}}])
    assert got[0].arguments == {"expression": "1+1"}
