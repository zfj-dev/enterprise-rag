"""工具抽象 + 内建工具（票 10）。一份实现、两种消费方式：进程内直调、经 MCP server 暴露。

每个工具自带 MCP 规范的 `name` / `description` / `inputSchema`，所以「可被发现」是白送的：
  进程内：`registry.specs()` / `registry.call(name, args)`
  对外  ：`app/mcp/server.py` 把同一份 specs 挂到 MCP server 上

`Calculator` 用 **AST 白名单**求值 —— **绝不用 eval/exec**。名字绑定、属性访问、下标、
导入、白名单外的函数调用，全部按构造拒绝（不是靠黑名单或转义）。
"""
from __future__ import annotations

import ast
import math

from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(RuntimeError):
    """工具入参不合法或执行失败 —— 上层（代理循环）据此降级，而不是抛穿。"""


# ---------- Calculator：AST 白名单求值 ----------

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b}
_UNARYOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_CMPOPS = {ast.Lt: lambda a, b: a < b, ast.Gt: lambda a, b: a > b,
           ast.LtE: lambda a, b: a <= b, ast.GtE: lambda a, b: a >= b,
           ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}

_MAX_POW = 64          # 拦 9**9**9 这种能把进程算死的幂
_MAX_ABS = 1e30        # 结果的上界，防连乘 / 幂运算爆掉
_MAX_DEPTH = 40        # 嵌套深度上限 —— 顺带把深嵌套引发的 RecursionError 挡在门外
_MAX_EXPR_CHARS = 500  # 表达式长度上限，同上


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("只支持数字（布尔、字符串、列表都不行）")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("结果不是有限数（NaN / Inf）")
    if abs(value) > _MAX_ABS:
        raise ToolError("数值太大，拒绝计算")
    return value


def _eval(node, depth: int = 0):
    """按白名单求值。**任何失败都转成 ToolError** —— 说好「不抛穿」就不能漏 TypeError/OverflowError。"""
    if depth > _MAX_DEPTH:
        raise ToolError("表达式嵌套太深")

    if isinstance(node, ast.Constant):
        return _num(node.value)

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolError("不支持的运算符：%s" % type(node.op).__name__)
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
            raise ToolError("幂次过大，拒绝计算")
        try:
            return _num(op(left, right))
        except ZeroDivisionError:
            raise ToolError("除数为零")
        except OverflowError:
            raise ToolError("数值溢出，拒绝计算")

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolError("不支持的一元运算符：%s" % type(node.op).__name__)
        return _num(op(_eval(node.operand, depth + 1)))

    if isinstance(node, ast.Compare):
        left = _eval(node.left, depth + 1)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _CMPOPS.get(type(op))
            if fn is None:
                raise ToolError("不支持的比较符：%s" % type(op).__name__)
            right = _eval(comparator, depth + 1)
            if not fn(left, right):
                return 0
            left = right
        return 1

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ToolError("只允许这几个函数：%s" % " / ".join(sorted(_FUNCS)))
        if node.keywords:
            raise ToolError("函数不支持关键字参数")
        args = [_eval(a, depth + 1) for a in node.args]
        try:
            return _num(_FUNCS[node.func.id](*args))
        except ToolError:
            raise
        except Exception as e:   # TypeError/ValueError/OverflowError… 都是「参数不对」，不是程序错
            raise ToolError("函数 %s 的参数不对：%s" % (node.func.id, e))

    # 到这儿说明碰到了白名单外的语法：名字绑定 / 属性访问 / 下标 / 导入 / 推导式 / lambda …
    raise ToolError("表达式里只允许数字与算术运算（不允许：%s）" % type(node).__name__)


def calculate(expression: str):
    """把一段算术表达式的值算出来。**不用 eval/exec** —— 只按上面的 AST 白名单走。

    任何问题（语法、越界、不可解析、算不动）都以 ToolError 抛出，绝不漏出别的异常类型。
    """
    text = str(expression or "").strip()
    if not text:
        raise ToolError("表达式为空")
    if len(text) > _MAX_EXPR_CHARS:
        raise ToolError("表达式太长（上限 %d 字符）" % _MAX_EXPR_CHARS)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ToolError("表达式语法错误：%s" % e.msg)
    except (ValueError, RecursionError) as e:      # 畸形输入也可能在这里炸
        raise ToolError("表达式无法解析：%s" % e)
    return _eval(tree.body)


# ---------- 工具抽象 ----------

@dataclass
class Tool:
    """一个工具：自带 MCP 规范元数据 + 一份实现。"""

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def spec(self) -> dict:
        """MCP 规范的工具定义（字段名照协议来，别改名）。"""
        return {"name": self.name, "description": self.description,
                "inputSchema": self.input_schema}

    def call(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """执行。入参不合法就抛 ToolError —— 由调用方决定降级方式。"""
        if not isinstance(arguments, (dict, type(None))):
            raise ToolError("参数必须是一个对象")
        return self.handler(arguments or {})


def _calc_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ToolError("缺少字符串参数 expression")
    return {"expression": expression, "value": calculate(expression)}


CALCULATOR = Tool(
    name="Calculator",
    description="计算一段算术表达式（+ - * / // % **、括号、比较；另支持 abs/round/min/max/sum）。"
                "只做算术，不碰名字、属性、下标与导入。",
    input_schema={
        "type": "object",
        "properties": {"expression": {"type": "string",
                                      "description": "要计算的算术表达式，例如 (12.5-10)/10*100"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calc_handler,
)


@dataclass
class ToolRegistry:
    """一组工具。进程内直调与对外暴露共用这同一份 —— 不写两套。"""

    tools: list[Tool] = field(default_factory=list)

    def get(self, name: str) -> Tool:
        for t in self.tools:
            if t.name == name:
                return t
        raise ToolError("没有这个工具：%s" % name)

    def specs(self) -> list[dict[str, Any]]:
        """MCP client 侧「发现」到的工具清单。"""
        return [t.spec() for t in self.tools]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.get(name).call(arguments)


def default_registry() -> ToolRegistry:
    """内建工具。KbRetrieve / SqlQuery 由票 12 / 13 补上。"""
    return ToolRegistry(tools=[CALCULATOR])
