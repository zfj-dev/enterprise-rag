"""单代理（ReAct）循环（票 11）+ 引用与 1 步自检（票 14）。
**不直接构造 LLM、不直接建 MCP 连接** —— 两者都从外部注入。

每一步：带工具定义问一次 LLM → 有工具调用就经传输执行、把结果回灌 → 再问；没有就是终答。

三条终止路径**都必须给出可返回的结果，绝不抛穿**：
  拿到终答 / 达步数上限 / 模型或工具出错。

终答再过 **1 步自检**（票 14）：没有依据就拒答，多步推理不是免检理由。
"""
from __future__ import annotations

import json
import logging
import time

from typing import Any

from app.core.citation import apply_no_source_no_claim, validate_sources, verify_claims
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


def _collect_sources(result: dict, seen_ids: set) -> list[dict]:
    """收工具带回来的来源，按 chunk_id 去重 —— 同一块被检索两次只算一次引用。"""
    out = []
    for s in result.get("sources") or []:
        if not isinstance(s, dict):
            continue
        cid = s.get("chunk_id")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        out.append(s)
    return out


def _self_check(answer: str, *, sources: list, retrieval_ran: bool,
                tool_grounded: bool, llm) -> tuple[str, dict]:
    """1 步自检（票 14）：**复用** citation 的既有实现，不另起一套。

    「no source → no claim」在代理链路同样成立，但**依据不止文档片段** ——
    Calculator / SqlQuery 成功给出的结果也是依据，算数题不该因为没引用就被拒。
    于是拒答只在"确实没拿到依据"时发生：
      · 检索类工具（结果带 sources）跑了却一无所获 —— **查了没查到，就是没依据**，
        中间夹多少次成功的别的工具都不解锁（否则模型自己插一次 Calculator 就能绕过拒答）；
      · 连一次成功的工具都没有，模型直接凭记忆作答（工具报错也算没拿到东西）。
    演示模式（Fake）不产生工具调用、等价一次普通生成，整段自检跳过
    （与 chat_service「真模型才校验引用」同口径；否则 demo 会恒拒答、与「退化为单步回答」冲突）。
    """
    trace: dict[str, Any] = {"self_check": "skipped", "citation_coverage": None}
    if getattr(llm, "is_fake", False):
        return answer, trace

    ccit = validate_sources(sources)
    if ccit.has_sources:
        trace["self_check"] = "passed_with_citation"
    elif tool_grounded and not retrieval_ran:
        trace["self_check"] = "passed_with_tool"
    else:                       # 查了没查到 / 连一次成功的工具都没有 —— 都没依据
        answer = apply_no_source_no_claim(answer, ccit)
        trace["self_check"] = "refused"
    try:
        # 无来源时 verify_claims 自己就返回 0，不会去调模型（与 chat_service 同一口径）
        trace["citation_coverage"] = verify_claims(answer, sources, llm).get("coverage")
    except Exception as e:      # noqa: BLE001 —— 自检炸了也不许把整轮回答丢掉
        logger.warning("引用覆盖率校验失败：%s", e)
    return answer, trace


def run_agent(question: str, *, llm, transport, max_steps: int = DEFAULT_MAX_STEPS,
              system: str | None = None) -> dict:
    """跑有上限的 ReAct 循环。返回 {answer, sources, steps, latency, stopped, trace}。

    steps 每步记 {tool, arguments, summary, ms, ok} —— 坏了能定位是选错工具还是工具本身错。
    latency 记 {total_ms, steps_ms} —— 多步的代价看得见（票 07 的分段口径）。
    trace 记 {self_check, citation_coverage} —— 这次到底有没有依据（票 14）：
    self_check ∈ skipped（Fake）/ refused（无依据，已拒答）/ passed_with_citation / passed_with_tool。
    """
    started = time.perf_counter()
    tools = transport.list_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system or _DEFAULT_SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[dict[str, Any]] = []
    sources: list[dict] = []
    seen_ids: set = set()
    # 检索类工具跑过 / 有没有**成功**的非检索依据（Calculator、SqlQuery）—— 拒答判定要用
    retrieval_ran = False
    tool_grounded = False
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
            # 结果带 sources 的算检索类；其余**成功**的（计算 / 元数据）算另一种依据 ——
            # 报错的工具什么也没给，不能算依据
            if isinstance(result.get("sources"), list):
                retrieval_ran = True
            elif not failed:
                tool_grounded = True
            sources.extend(_collect_sources(result, seen_ids))
            messages.append(_tool_message(call, result))

    if stopped != "answered":
        # 没拿到终答（跑满上限 / 模型出错）也要收敛：再问一次但**不给工具**，逼它用手里的信息作答
        try:
            answer = llm.chat_with_tools(messages, tools=None).get("content") or answer
        except Exception as e:             # noqa: BLE001 —— 收敛这一步也失败，就把已有的返回
            logger.warning("收敛作答失败：%s", e)

    answer, trace = _self_check(answer, sources=sources, retrieval_ran=retrieval_ran,
                                tool_grounded=tool_grounded, llm=llm)

    return {
        "answer": answer,
        "sources": sources,
        "steps": steps,
        "latency": {"total_ms": round((time.perf_counter() - started) * 1000, 1),
                    "steps_ms": [s["ms"] for s in steps]},
        "stopped": stopped,
        "trace": trace,
    }
