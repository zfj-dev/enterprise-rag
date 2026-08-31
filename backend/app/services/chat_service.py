"""问答管线：问题优化 → 混合检索 → 引用校验 → Prompt → LLM 生成 → 持久化会话/消息/反馈。"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Callable, Iterator

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


# ---------- 泛化"枚举某类内容"（表/图/公式/代码…） ----------
# ---------- 泛化"枚举某类内容"（表/图/公式/代码…） ----------
_ENUM_SIGNALS = ("所有", "全部", "列出", "列举", "有哪些", "哪些", "都有", "给出", "给我", "汇总", "统计")


def _looks_like_table(content: str) -> bool:
    lines = (content or "").splitlines()
    if sum(1 for ln in lines if ln.lstrip().startswith("|")) >= 2:
        return True
    return bool(re.match(r"^(表|Table)\s*\d", (content or "").strip(), re.IGNORECASE))


def _looks_like_figure(content: str) -> bool:
    c = content or ""
    if "<!-- image -->" in c or "<!-- figure" in c.lower():
        return True
    return bool(re.match(r"^(图|Fig)", c.strip(), re.IGNORECASE))


def _looks_like_formula(content: str) -> bool:
    c = content or ""
    if "formula" in c.lower() and "<!--" in c:
        return True
    return bool(re.search(r"\$|\\begin\{|\\(", c))


def _looks_like_code(content: str) -> bool:
    c = content or ""
    return "```" in c or "<!-- code" in c.lower()


# (类别名, 查询关键词正则, 具体编号排除正则, chunk 判定函数)
_TYPE_RULES: list[tuple[str, "re.Pattern", "re.Pattern | None", Callable[[str], bool]]] = [
    ("table",
     re.compile(r"表|表格|table", re.IGNORECASE),
     re.compile(r"(表|表格)\s*\d|table\s*\d", re.IGNORECASE),
     _looks_like_table),
    ("figure",
     re.compile(r"图|图片|图像|figure|fig", re.IGNORECASE),
     re.compile(r"图\s*\d|figure\s*\d", re.IGNORECASE),
     _looks_like_figure),
    ("formula",
     re.compile(r"公式|equation|formula", re.IGNORECASE),
     re.compile(r"公式\s*[（(]?\s*\d|equation\s*\d", re.IGNORECASE),
     _looks_like_formula),
    ("code",
     re.compile(r"代码|code|程序", re.IGNORECASE),
     None,
     _looks_like_code),
]


def _enum_intent(question: str) -> str | None:
    """检测"枚举某类内容"意图，返回类别名（table/figure/…），否则 None。

    规则：问题含枚举信号词（所有/列出/有哪些/汇总…）且命中某类别关键词，
    但不是具体编号引用（表3.1/图2.1）→ 属于具体引用，走普通检索。
    """
    q = question or ""
    if not any(s in q for s in _ENUM_SIGNALS):
        return None
    for name, kw_re, specific_re, _ in _TYPE_RULES:
        if kw_re.search(q) and not (specific_re and specific_re.search(q)):
            return name
    return None


def _classifier_for(category: str) -> Callable[[str], bool]:
    for name, _, _, classifier in _TYPE_RULES:
        if name == category:
            return classifier
    return lambda c: False


def _load_typed_chunks(db: Session, kb_id: str, owner_id: str, classifier: Callable[[str], bool], limit: int = 40) -> list[dict]:
    """从库里取出命中某分类器的 child chunk，按页序，最多 limit 条。"""
    from app.models.entities import Chunk, Document

    rows = (db.query(Chunk, Document.filename)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.kb_id == kb_id, Chunk.owner_id == owner_id,
                    Chunk.chunk_type == "child")
            .order_by(Chunk.page_num.asc(), Chunk.id.asc())
            .all())
    out: list[dict] = []
    for r, doc_name in rows:
        if not classifier(r.content):
            continue
        out.append({
            "chunk_id": r.id,
            "content": r.content,
            "parent_content": r.parent_content,
            "metadata": {
                "kb_id": kb_id, "owner_id": owner_id, "doc_name": doc_name or "未知文档",
                "page_num": r.page_num, "content": r.content,
            },
            "rank_score": 1.0,
        })
        if len(out) >= limit:
            break
    return out


def _merge_typed_candidates(base: list[dict], typed: list[dict], max_total: int = 50) -> list[dict]:
    seen = {c["chunk_id"] for c in base}
    merged = list(base)
    for t in typed:
        if t["chunk_id"] not in seen:
            seen.add(t["chunk_id"])
            merged.append(t)
    return merged[:max_total]


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
    cat = _enum_intent(question)
    # 枚举类问题本身已完整清晰：跳过 LLM 改写，避免历史污染查询
    q2 = question if cat else _maybe_rewrite(rt, question, history)
    candidates = rt.retriever.retrieve(q2, kb_id=kb_id, owner_id=user.id)
    enum_hint = None
    if cat:
        typed = _load_typed_chunks(db, kb_id, user.id, _classifier_for(cat))
        if typed:
            candidates = _merge_typed_candidates(candidates, typed)
        cat_names = {"table": "表格", "figure": "图片/图", "formula": "公式", "code": "代码"}
        enum_hint = (f"当前问题要求枚举/列出所有【{cat_names.get(cat, cat)}】。"
                     f"请严格依据【参考资料】中对应类型的块，逐一列出其编号、标题和完整内容，不要遗漏；"
                     f"历史对话中的列表仅作上下文参考，不要重复其中的其他类型内容。")
    ccit = validate_sources(candidates)
    context = format_context(candidates)
    prompt = build_prompt(question, context, history=history, enum_hint=enum_hint)

    user_msg = ChatMessage(session_id=sess.id, role="user", content=question)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    if not sess.title:  # 自动标题：取首个问题前 30 字
        sess.title = question.strip().replace("\n", " ")[:30] or "新对话"
        db.commit()

    return Prep(
        session_id=sess.id, user=user, kb_id=kb_id, question=question, rewrite=q2,
        candidates=candidates, sources=_to_sources(candidates), prompt=prompt, _ccit=ccit,
        trace={"query": question, "rewrite": q2, "history_turns": len(history),
               "enum_cat": cat, "candidates": len(candidates),
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
