"""问答管线：问题优化 → 混合检索 → 引用校验 → Prompt → LLM 生成 → 持久化会话/消息/反馈。"""
from __future__ import annotations

import re
import time

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
        # 前端以 UUID(v4) 作为 conversationId 直接当会话 id，以便多对话互相隔离、多轮上下文对得上
        sess = ChatSession(id=session_id, user_id=user.id, kb_id=kb_id, title="")
        db.add(sess)
        db.commit()
        db.refresh(sess)
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
    return bool(re.search(r"\$|\\begin\{|\\\(", c))


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


def _named_ref_intent(question: str) -> list[tuple[str, str, str | None]]:
    """检测具体编号引用（表3.2 / 图 3 . 1 / 公式1），返回 [(前缀, 主号, 副号)]。"""
    out = []
    for m in re.finditer(r"(表|表格|图|公式)\s*(\d+)\s*(?:[.\-]\s*(\d+))?", question or ""):
        out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _load_named_chunks(db: Session, kb_id: str, owner_id: str, prefix: str, n1: str, n2: str | None, limit: int = 3) -> list[dict]:
    """按具体编号引用（如表3.2）取 chunk：优先表格（含 | 行），其次含引用的句子。"""
    from app.models.entities import Chunk, Document
    key = f"{prefix}{n1}.{n2}" if n2 else f"{prefix}{n1}"
    rows = (db.query(Chunk, Document.filename)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.kb_id == kb_id, Chunk.owner_id == owner_id, Chunk.chunk_type == "child")
            .all())
    tables: list[dict] = []
    refs: list[dict] = []
    for r, doc_name in rows:
        compact = re.sub(r"\s+", "", r.content or "")
        if key not in compact:
            continue
        entry = {
            "chunk_id": r.id, "content": r.content, "parent_content": r.parent_content,
            "metadata": {"kb_id": kb_id, "owner_id": owner_id, "doc_name": doc_name or "未知文档",
                         "page_num": r.page_num, "content": r.content},
            "rank_score": 2.0,
        }
        (tables if ("|" in r.content or _looks_like_table(r.content)) else refs).append(entry)
    return (tables[:2] + refs[:1])[:limit]


def _prepend_named(base: list[dict], named: list[dict], max_total: int = 20) -> list[dict]:
    """把按编号命中的 chunk 前置到候选首位（LLM 优先看到正确表格）。"""
    seen = {c["chunk_id"] for c in named}
    merged = list(named)
    for c in base:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            merged.append(c)
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
    refs = [r for r in _named_ref_intent(question) if r[0] in ("表", "表格", "图", "公式")]
    # 枚举/具体编号类问题本身完整清晰：跳过 LLM 改写，避免历史污染查询
    q2 = question if (cat or refs) else _maybe_rewrite(rt, question, history)
    candidates = rt.retriever.retrieve(q2, kb_id=kb_id, owner_id=user.id)
    enum_hint = None
    if cat:
        typed = _load_typed_chunks(db, kb_id, user.id, _classifier_for(cat))
        if typed:
            # 枚举：typed 块按文档顺序(页码, 块内id)前置，其余检索候选缀后，引导 LLM 按"出现顺序"列
            typed_sorted = sorted(typed, key=lambda c: (c["metadata"].get("page_num", 0), str(c.get("chunk_id", ""))))
            in_typed = {t["chunk_id"] for t in typed}
            candidates = typed_sorted + [c for c in candidates if c.get("chunk_id") not in in_typed]
        cat_names = {"table": "表格", "figure": "图片/图", "formula": "公式", "code": "代码"}
        if cat == "formula":
            enum_hint = ("当前问题要求枚举/列出所有【公式】。"
                         "请严格按照【参考资料】中公式出现的先后顺序（第X页从小到大、同页按块顺序）从前往后逐一列出，每条标注所在【第X页】；"
                         "凡资料中出现的公式都要列出，不要遗漏，也不要因为某条公式看起来被截断/不完整就跳过——把它当作公式候选并标注页码；"
                         "每条公式请输出为**规范、紧凑的标准 LaTeX**：去掉多余空格、把字母间距合并（如 `S m o o t h`→`Smooth`、`\\frac { 1 } { N }`→`\\frac{1}{N}`）、"
                         "修正明显可辨的 OCR 误读（如通道注意力应写 `M_c` 而非 `L_c`、空间注意力应写 `M_s` 而非 `O_s`、`\\textcircled{=}0.5`→`@0.5`、`\\frac{1}{c}`→`\\frac{1}{C}`）；公式内不要用 Markdown 链接/mailto 语法，直接写符号（如写 `mAP@0.5`，不要 `[mAP@0.5](mailto:...)`）；"
                         "但**不得改变数学含义、不得臆造公式**：对确被截断的公式只保留能读到的部分、绝不补全下半截；不过若识别结果与上下文/标准定义明显不符（如漏了 `L_{IoU}`、`L_{CE}` 这类左侧标识符，或中间环节如 `= 1 - IoU =` 缺失），可依据上下文与标准数学定义把**残缺处修正为标准写法**；"
                         "若个别字符前后文不足以确定，宁可保留原样也不要编造；"
                         "数量与列出条目一一对应，不确定不要输出数量；历史对话中的公式列表仅作上下文参考，不要重复其中的非公式内容。"
                         "每条公式后用 [来源: 文档名, 第X页] 标注来源（文档名用【参考资料】里的文件名，如 [来源: 毕设论文.pdf, 第18页]），不要只写 [第X页]。")
        else:
            enum_hint = (f"当前问题要求枚举/列出所有【{cat_names.get(cat, cat)}】。"
                         f"请严格依据【参考资料】中对应类型的块，按出现顺序（第X页从小到大）逐一列出其编号（如 表3.1、表3.2、表4.1 等）、标题和完整内容，"
                         f"凡是以'表'+数字编号的都要包含，不要因任何理由遗漏；"
                         f"若提到数量，数量必须与列出的条目一一对应，不确定时不要输出数量；"
                         f"历史对话中的列表仅作上下文参考，不要重复其中的其他类型内容。")
    # 具体编号引用（表3.2/图3.1）：强制注入含该编号的 chunk 并前置，避免"如表3.3所示"等句子抢位
    if refs:
        for prefix, n1, n2 in refs:
            named = _load_named_chunks(db, kb_id, user.id, prefix, n1, n2)
            if named:
                candidates = _prepend_named(candidates, named)
        # 只保留含该编号的候选，避免 LLM 被其他表格/段落带偏（选错表）
        keys = [f"{r[0]}{r[1]}{'.' + r[2] if r[2] else ''}" for r in refs]
        candidates = [c for c in candidates
                      if any(k in re.sub(r"\s+", "", c.get("content", "") or "") for k in keys)]
        ref_str = ", ".join(keys)
        enum_hint = (f"当前问题要求的是【{ref_str}】的具体内容。"
                     f"请严格依据【参考资料】中该编号对应的块回答，不要引用历史对话中的其他表格/图片/内容。")
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
    # 精确编号引用（表3.1 vs 表3.3 只差数字）跳过语义缓存，避免返回相似问题的旧答案
    precise = bool(_named_ref_intent(prep.question))
    use_cache = get_settings().semantic_cache and not precise
    cache_hit = False
    hit = None
    if use_cache:
        hit = rt.semantic_cache.get(prep.question, prep.kb_id)
        if hit:
            cache_hit = True
    if cache_hit:
        # 缓存命中：仍走"流式外观"——把答案切成小块逐条下发，前端逐段追加；禁止整块一次性插入
        answer = apply_no_source_no_claim(hit["answer"], prep._ccit)
        for i in range(0, len(answer), 12):
            yield {"type": "delta", "text": answer[i:i + 12]}
            time.sleep(0.01)
    else:
        chunks: list[str] = []
        for piece in rt.llm.stream([{"role": "user", "content": prep.prompt}]):
            chunks.append(piece)
            yield {"type": "delta", "text": piece}
        if use_cache:
            rt.semantic_cache.put(prep.question, prep.kb_id, "".join(chunks))
        answer = apply_no_source_no_claim("".join(chunks), prep._ccit)

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
