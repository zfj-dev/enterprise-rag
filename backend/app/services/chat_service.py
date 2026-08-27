"""问答管线：问题优化 → 混合检索 → 引用校验 → Prompt → LLM 生成 → 持久化会话/消息/反馈。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.citation import apply_no_source_no_claim, validate_sources
from app.core.container import Runtime
from app.core.prompt import build_prompt, build_rewrite_prompt, format_context
from app.models.entities import ChatMessage, ChatSession, User


@dataclass
class Prep:
    session_id: str
    user: User
    kb_id: str
    question: str
    rewrite: str | None
    candidates: list[dict]
    sources: list[dict]
    prompt: str
    _ccit: object = None

    trace: dict = field(default_factory=dict)


def _maybe_rewrite(rt: Runtime, question: str, history: list[dict]) -> str:
    if get_settings().llm_provider == "fake":
        return question  # 演示/测试不真调 LLM，保持确定性；真实模式才做"问题优化"
    messages = [{"role": "user", "content": build_rewrite_prompt(question, history)}]
    rewritten = "".join(rt.llm.stream(messages)).strip()
    return rewritten or question


def _get_or_create_session(db: Session, user: User, kb_id: str, session_id: str | None) -> ChatSession:
    if session_id:
        sess = db.get(ChatSession, session_id)
        if sess:
            return sess
    sess = ChatSession(user_id=user.id, kb_id=kb_id, title="")
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def _to_sources(candidates: list[dict]) -> list[dict]:
    out = []
    for c in candidates:
        md = c.get("metadata", {}) or {}
        out.append({
            "chunk_id": c["chunk_id"],
            "doc_name": md.get("doc_name", "未知文档"),
            "page": md.get("page_num", 0),
            "text": c.get("content", ""),
            "score": float(c.get("rank_score", c.get("rrf_score", 0.0))),
        })
    return out


def prepare(db: Session, rt: Runtime, user: User, kb_id: str, question: str,
            session_id: str | None = None) -> Prep:
    q2 = _maybe_rewrite(rt, question, [])
    candidates = rt.retriever.retrieve(q2, kb_id=kb_id, owner_id=user.id)
    ccit = validate_sources(candidates)
    context = format_context(candidates)
    prompt = build_prompt(question, context)
    sess = _get_or_create_session(db, user, kb_id, session_id)

    user_msg = ChatMessage(session_id=sess.id, role="user", content=question)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    return Prep(
        session_id=sess.id, user=user, kb_id=kb_id, question=question, rewrite=q2,
        candidates=candidates, sources=_to_sources(candidates), prompt=prompt, _ccit=ccit,
        trace={"query": question, "rewrite": q2,
               "retrieval_top": candidates[:5], "sources_usable": ccit.has_sources},
    )


def stream_answer(db: Session, rt: Runtime, prep: Prep) -> Iterator[dict]:
    """流式生成：先给 sources，再增量给 answer，最后 done。"""
    yield {"type": "sources", "data": prep.sources}
    chunks: list[str] = []
    for piece in rt.llm.stream([{"role": "user", "content": prep.prompt}]):
        chunks.append(piece)
        yield {"type": "delta", "text": piece}
    answer = "".join(chunks)
    answer = apply_no_source_no_claim(answer, prep._ccit)

    asst = ChatMessage(session_id=prep.session_id, role="assistant", content=answer)
    db.add(asst)
    db.commit()
    db.refresh(asst)

    prep.trace["message_id"] = asst.id
    yield {"type": "done", "message_id": asst.id, "sources": prep.sources, "answer": answer}


def answer(db: Session, rt: Runtime, user: User, kb_id: str, question: str,
           session_id: str | None = None) -> dict:
    """同步问答（配合非流式/测试）。返回 {session_id, answer, sources, message_id, trace}。"""
    prep = prepare(db, rt, user, kb_id, question, session_id)
    result: dict = {}
    for ev in stream_answer(db, rt, prep):
        result = ev
    return {
        "session_id": prep.session_id,
        "answer": result.get("answer", ""),
        "sources": prep.sources,
        "message_id": result.get("message_id"),
        "trace": prep.trace,
    }
