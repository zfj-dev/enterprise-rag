"""工具层 + Calculator（票 10 / #17）：AST 白名单求值，绝不用 eval/exec。

传输在测试里走**进程内**直调 —— 不起真实 MCP 进程、不联网（spec 0002 的要求）。
真机往返（真起 stdio server）由 backend/verify_mcp.py 单独验。
"""
from __future__ import annotations

import json

import pytest

from app.core.tools import CALCULATOR, ToolError, calculate, default_registry
from app.mcp.client import InProcessTransport


# ---------- Calculator：该算对的算对 ----------

@pytest.mark.parametrize("expression, expected", [
    ("(12.5-10)/10*100", 25.0),
    ("2**10", 1024),
    ("-3 + +4", 1),
    ("7 // 2", 3),
    ("7 % 2", 1),
    ("abs(-3) + round(2.6)", 6),
    ("min(1, 2, 3)", 1),
    ("1 < 2", 1),
    ("2 < 1", 0),
    ("1 <= 1 < 2", 1),
])
def test_calculator_computes(expression, expected):
    assert calculate(expression) == expected


def test_calculator_trims_whitespace():
    assert calculate("  1 + 2  ") == 3


# ---------- Calculator：不信任模型的输入 ----------

@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",   # 导入
    "(1).__class__",                        # 属性访问
    "data[0]",                              # 下标
    "x + 1",                                # 名字绑定
    "lambda: 1",
    "[i for i in range(3)]",
    "open('/etc/passwd')",                  # 白名单外的函数
    "print(1)",
    "abs(x=1)",                             # 关键字参数
    "9**9**9",                              # 幂次过大
    "1/0",                                  # 除零
    "True",                                 # 布尔不是数字
    "'abc'",
    "",
    "1 +",                                  # 语法错误
])
def test_calculator_rejects_everything_that_is_not_arithmetic(expression):
    with pytest.raises(ToolError):
        calculate(expression)


def test_calculator_never_uses_eval_or_exec():
    """按构造杜绝注入 —— 源码里不该有对内置 eval/exec/compile 的调用。

    查的是 AST 里的**真实调用**，不是文本里有没有那四个字母
    （本模块自己的求值函数就叫 `_eval`，文本比对会把它误伤）。
    """
    import ast
    import inspect

    from app.core import tools as tools_mod

    called = {n.func.id for n in ast.walk(ast.parse(inspect.getsource(tools_mod)))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not ({"eval", "exec", "compile"} & called)


# ---------- 工具元数据（可发现性） ----------

def test_tools_carry_mcp_shaped_metadata():
    specs = default_registry().specs()
    assert [s["name"] for s in specs] == ["Calculator"]

    spec = specs[0]
    assert set(spec) == {"name", "description", "inputSchema"}    # 字段名照 MCP 协议来
    assert spec["description"]
    schema = spec["inputSchema"]
    assert schema["type"] == "object"
    assert "expression" in schema["properties"]
    assert schema["required"] == ["expression"]


def test_registry_calls_by_name():
    assert default_registry().call("Calculator", {"expression": "1+1"})["value"] == 2


def test_unknown_tool_is_an_error_not_a_crash():
    with pytest.raises(ToolError):
        default_registry().call("NoSuchTool", {})


@pytest.mark.parametrize("arguments", [
    {},                       # 缺参数
    {"expression": 42},       # 类型不对
    "不是对象",
])
def test_bad_arguments_raise_tool_error(arguments):
    with pytest.raises(ToolError):
        CALCULATOR.call(arguments)


# ---------- 进程内传输：工具只实现一份 ----------

def test_in_process_transport_exposes_the_same_registry():
    transport = InProcessTransport(default_registry())

    assert [t["name"] for t in transport.list_tools()] == ["Calculator"]
    assert transport.call_tool("Calculator", {"expression": "6*7"})["value"] == 42


def test_in_process_transport_propagates_tool_errors():
    transport = InProcessTransport(default_registry())
    with pytest.raises(ToolError):
        transport.call_tool("Calculator", {"expression": "data[0]"})


# ---------- 求值器不能漏出别的异常类型 ----------

@pytest.mark.parametrize("expression", [
    "min()",                          # 参数不对 → TypeError
    "sum(1, 2, 3)",
    "round(1, 2, 3)",
    "1e15**63",                       # float 幂溢出 → OverflowError
    "1" + "+1" * 1000,                # 太长
    "1" + "+(1" * 60 + ")" * 60,      # 嵌套太深 → RecursionError
])
def test_calculator_never_leaks_a_non_tool_error(expression):
    """说好「不抛穿」就不能漏 TypeError / OverflowError / RecursionError —— 只准抛 ToolError。"""
    with pytest.raises(ToolError):
        calculate(expression)


# ---------- MCP server 侧的应答（摊成纯函数，不起真进程） ----------

def test_server_lists_tools_with_their_schema():
    pytest.importorskip("mcp.types", reason="需要 mcp SDK（见 requirements-real.txt）")
    from app.mcp.server import list_tools_result

    got = list_tools_result(default_registry())
    assert [t.name for t in got.tools] == ["Calculator"]
    assert got.tools[0].input_schema["required"] == ["expression"]


def test_server_returns_a_failed_result_with_the_reason():
    pytest.importorskip("mcp.types", reason="需要 mcp SDK（见 requirements-real.txt）")
    from app.mcp.server import call_tool_result

    ok = call_tool_result(default_registry(), "Calculator", {"expression": "(12.5-10)/10*100"})
    assert ok.is_error is False
    assert json.loads(ok.content[0].text)["value"] == 25.0

    bad = call_tool_result(default_registry(), "Calculator", {"expression": "data[0]"})
    assert bad.is_error is True
    assert "只允许数字与算术运算" in json.loads(bad.content[0].text)["error"]


# ---------- 客户端把应答摊平 ----------

def test_result_to_dict_keeps_the_error_reason():
    """失败时不能拿布尔把原因冲掉 —— 调用方得知道为什么失败。"""
    from app.mcp.client import _result_to_dict

    class _Text:
        type = "text"
        text = '{"error": "表达式里只允许数字与算术运算"}'

    class _Result:
        content = [_Text()]
        is_error = True
        structured_content = None

    got = _result_to_dict(_Result())
    assert got["is_error"] is True
    assert "只允许数字" in got["error"]


def test_result_to_dict_prefers_structured_content():
    from app.mcp.client import _result_to_dict

    class _Result:
        content = []
        is_error = False
        structured_content = {"value": 42}

    assert _result_to_dict(_Result()) == {"value": 42}
