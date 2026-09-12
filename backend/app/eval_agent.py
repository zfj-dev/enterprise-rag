"""把两条问答链路包成评测核心要的 answer_fn（票 15）。

评测核心（票 01）只认一个缝：「问题 -> 答案 + 来源」。代理链路接进来**不改评测代码**，
确定性链路也走同一个缝 —— 同形才谈得上对比。

- `agent_answer_fn`        ReAct 代理（票 11-14）
- `deterministic_answer_fn` 现有单步管线（chat_service.answer）

llm 与 transport 都可注入：单测传脚本化 stub，不起真实 MCP 进程、不联网。
"""
from __future__ import annotations

from app.config import get_settings
from app.core.agent import run_agent
from app.eval_core import AnswerFn
from app.mcp.client import InProcessTransport
from app.mcp.registry import build_registry


def _as_answer(answer: str, sources: list, coverage, context=None) -> dict:
    """两侧共用的出参形状 —— 少一个键，评测核心就少算一项。

    `context` 是压缩前/后 token 与口径（票 19）：代理链路不装配 prepare 那份上下文，
    所以它没有这个数（报告里会如实写「不可用」，不编）。
    """
    return {"answer": answer or "", "sources": sources or [],
            "citation_coverage": coverage, "context": context}


def agent_answer_fn(db, rt, user, kb_id: str, *, max_steps: int | None = None,
                    llm=None, transport=None) -> AnswerFn:
    """代理链路 -> answer_fn。工具集按**本次请求的上下文**造（范围服务端注入）。

    步数上限与 LLM 都和生产链路同源（`rt.llm_for` 就是票 31 的按用户解析）——
    评测量到的该是线上跑的那条链路，不是另配一条。
    """
    max_steps = max_steps or get_settings().agent_max_steps
    llm = llm or rt.llm_for(user.id)
    transport = transport or InProcessTransport(build_registry(db, rt, user, kb_id))

    def ask(question: str) -> dict:
        # 压缩策略与线上链路同口径（票 21）：关压缩就不收，枚举/编号查询走同一条豁免 ——
        # 否则代理这一列会无条件收缩，与确定性那一列不可比（尤其是"列全"类问题）。
        from app.services.chat_service import compress_exempt

        got = run_agent(question, llm=llm, transport=transport, max_steps=max_steps,
                        trim_tool_results=get_settings().context_compress
                        and not compress_exempt(question))
        return _as_answer(got["answer"], got["sources"], got["trace"].get("citation_coverage"))

    return ask


def deterministic_answer_fn(db, rt, user, kb_id: str, session_id: str | None = None) -> AnswerFn:
    """确定性单步链路 -> answer_fn（同形，供对比）。

    **显式关掉代理**：对比的基准链路不能被全局开关顺手换成代理，否则两列都是代理、对比是假的。
    """
    from app.services import chat_service

    def ask(question: str) -> dict:
        out = chat_service.answer(db, rt, user, kb_id, question, session_id, allow_agent=False)
        trace = out.get("trace") or {}
        return _as_answer(out.get("answer"), out.get("sources"), trace.get("citation_coverage"),
                          trace.get("context_tokens"))

    return ask
