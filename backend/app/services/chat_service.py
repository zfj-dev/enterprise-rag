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


def _load_history(db: Session, session_id: str, limit: int = 6) -> list[dict]:
    """取最近几轮的 {user, assistant} 对，用于多轮指代消解与上下文。"""
    msgs = (db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc()).limit(limit).all())
    msgs = list(reversed(msgs))
    pairs: list[dict] = []
    cur: dict | None = None
    for m in msgs:
        if m.role == "user":
            if cur:
                pairs.append(cur)
            cur = {"user": m.content, "assistant": ""}
        elif m.role == "assistant" and cur is not None:
            cur["assistant"] = m.content
    if cur:
        pairs.append(cur)
    return pairs


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
    sess = _get_or_create_session(db, user, kb_id, session_id)
    history = _load_history(db, sess.id, limit=6)
    q2 = _maybe_rewrite(rt, question, history)
    candidates = rt.retriever.retrieve(q2, kb_id=kb_id, owner_id=user.id)
    ccit = validate_sources(candidates)
    context = format_context(candidates)
    prompt = build_prompt(question, context, history=history)

    user_msg = ChatMessage(session_id=sess.id, role="user", content=question)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    return Prep(
        session_id=sess.id, user=user, kb_id=kb_id, question=question, rewrite=q2,
        candidates=candidates, sources=_to_sources(candidates), prompt=prompt, _ccit=ccit,
        trace={"query": question, "rewrite": q2, "history_turns": len(history),
               "retrieval_top": candidates[:5], "sources_usable": ccit.has_sources},
    )


def stream_answer(db: Session, rt: Runtime, prep: Prep) -> Iterator[dict]:
    """流式生成：先给 sources，再增量给 answer，最后 done。含语义缓存与引用覆盖率。"""
    yield {"type": "sources", "session_id": prep.session_id, "data": prep.sources}
    cache_hit = False
    if get_settings().semantic_cache:
        hit = rt.semantic_cache.get(prep.question, prep.kb_id)
        if hit:
            cache_hit = True
            yield {"type": "delta", "text": hit["answer"]}
    if not cache_hit:
        chunks: list[str] = []
        for piece in rt.llm.stream([{"role": "user", "content": prep.prompt}]):
            chunks.append(piece)
            yield {"type": "delta", "text": piece}
        if get_settings().semantic_cache:
            rt.semantic_cache.put(prep.question, prep.kb_id, "".join(chunks))
    answer = "".join(chunks) if not cache_hit else hit["answer"]
    answer = apply_no_source_no_claim(answer, prep._ccit)

    if get_settings().llm_provider != "fake":  # 真实模式：逐句校验引用覆盖率
        try:
            from app.core.citation import verify_claims
            cov = verify_claims(answer, prep.sources, rt.llm)
            prep.trace["citation_coverage"] = cov.get("coverage")
        except Exception:
            pass

    asst = ChatMessage(session_id=prep.session_id, role="assistant", content=answer)
    db.add(asst)
    db.commit()
    db.refresh(asst)

    prep.trace["message_id"] = asst.id
    prep.trace["cache_hit"] = cache_hit
    yield {"type": "done", "session_id": prep.session_id, "message_id": asst.id, "sources": prep.sources,
           "answer": answer, "cache_hit": cache_hit,
           "citation_coverage": prep.trace.get("citation_coverage")}


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
