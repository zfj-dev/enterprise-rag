"""单代理（ReAct）循环（票 11）。**不直接构造 LLM、不直接建 MCP 连接** —— 两者都从外部注入。

每一步：带工具定义问一次 LLM → 有工具调用就经传输执行、把结果回灌 → 再问；没有就是终答。

三条终止路径**都必须给出可返回的结果，绝不抛穿**：
  拿到终答 / 达步数上限 / 模型或工具出错。
"""
from __future__ import annotations

import json
import logging
import time

from typing import Any

from app.core.llm import encode_tool_calls
from app.utils.text import truncate

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4
_SUMMARY_CHARS = 300

_DEFAULT_SYSTEM = (
    "你可以调用工具来解决问题：需要精确计算、查资料或查元数据时，优先用工具，不要凭记忆心算。"
    "拿到足够信息后直接给出答案；工具报错就换个办法或如实说明，不要空转。"
)


def _tool_message(call, result: dict) -> dict:
    """把工具结果按 OpenAI 兼容的 tool 消息回灌 —— 模型据此才知道工具给了什么（含错误原因）。"""
    return {"role": "tool", "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False)}


def _assistant_message(content: str, calls: list) -> dict:
    """把模型这一轮的回复（含它要求的工具调用）按协议回灌，否则下一轮上下文对不上。"""
    return {"role": "assistant", "content": content or "",
            "tool_calls": encode_tool_calls(calls)}


def _summarize(result: dict) -> str:
    text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
    return truncate(text, _SUMMARY_CHARS)


def run_agent(question: str, *, llm, transport, max_steps: int = DEFAULT_MAX_STEPS,
              system: str | None = None) -> dict:
    """跑有上限的 ReAct 循环。返回 {answer, sources, steps, latency, stopped}。

    steps 每步记 {tool, arguments, summary, ms, ok} —— 坏了能定位是选错工具还是工具本身错。
    latency 记 {total_ms, steps_ms} —— 多步的代价看得见（票 07 的分段口径）。
    """
    started = time.perf_counter()
    tools = transport.list_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system or _DEFAULT_SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[dict[str, Any]] = []
    sources: list[dict] = []
    answer = ""
    stopped = "max_steps"

    for _ in range(max(1, max_steps)):
        try:
            outcome = llm.chat_with_tools(messages, tools=tools)
        except Exception as e:      # noqa: BLE001 —— 模型这一步出错也不能把整轮丢掉
            logger.warning("代理第 %d 步的模型调用失败：%s", len(steps) + 1, e)
            stopped = "llm_error"
            break

        calls = list(outcome.get("tool_calls") or [])
        answer = outcome.get("content") or answer

        if not calls:                      # 没有工具调用了 —— 这就是终答
            stopped = "answered"
            break

        messages.append(_assistant_message(outcome.get("content") or "", calls))
        for call in calls:
            began = time.perf_counter()
            try:
                result = transport.call_tool(call.name, call.arguments) or {}
            except Exception as e:         # noqa: BLE001 —— 单次工具失败不丢整个回答
                result = {"error": "%s: %s" % (type(e).__name__, e)}
                logger.warning("工具 %s 调用失败（已降级继续）：%s", call.name, e)
            ms = (time.perf_counter() - began) * 1000

            failed = bool(result.get("is_error")) or "error" in result
            steps.append({"tool": call.name, "arguments": call.arguments,
                          "summary": _summarize(result), "ms": round(ms, 1), "ok": not failed})
            got_sources = result.get("sources")
            if isinstance(got_sources, list):
                sources.extend(s for s in got_sources if isinstance(s, dict))
            messages.append(_tool_message(call, result))

    if stopped != "answered":
        # 没拿到终答（跑满上限 / 模型出错）也要收敛：再问一次但**不给工具**，逼它用手里的信息作答
        try:
            answer = llm.chat_with_tools(messages, tools=None).get("content") or answer
        except Exception as e:             # noqa: BLE001 —— 收敛这一步也失败，就把已有的返回
            logger.warning("收敛作答失败：%s", e)

    return {
        "answer": answer,
        "sources": sources,
        "steps": steps,
        "latency": {"total_ms": round((time.perf_counter() - started) * 1000, 1),
                    "steps_ms": [s["ms"] for s in steps]},
        "stopped": stopped,
    }
