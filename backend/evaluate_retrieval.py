"""离线检索质量评估：量化「检索层」是否把含答案的 chunk 捞进 top-k。

指标本身在可注入的评测核心 app/eval_core.py（run_retrieval_eval）；本脚本只做两件事：
把索引与四种检索方式包成 retrieve_fn、把核心算出的指标落盘。
**不依赖运行中的服务、不需要 LLM。**

用法:  cd backend && python evaluate_retrieval.py
报告:  logs/retrieval-eval-report.log
环境:  EMBEDDING_PROVIDER=bge 时才是真质量(FakeEmbedding 仅演示管线)。
"""
from __future__ import annotations

import json
import os
import time

from app.eval_core import embedding_label, normalize, run_retrieval_eval
from app.eval_index import index_chunks

BACKEND = os.path.dirname(os.path.abspath(__file__))

REPORT = os.path.join(BACKEND, "logs", "retrieval-eval-report.log")
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_retrieval.json"))
KB_ID, OWNER_ID = "eval_kb", "eval_owner"
K_LIST = (3, 5, 10)  # 评估 hit-rate/recall/MRR 的 top-k


def _index_doc(rt, path: str) -> list[dict]:
    """解析→按页分块→只取 child→向量+BM25 入库。返回 child 块列表(含 id/content/page/type)。"""
    parsed = rt.parser.parse(path, os.path.basename(path))
    if parsed.metadata.get("error"):
        raise RuntimeError(f"解析失败: {parsed.metadata['error']}")
    page_texts = parsed.pages if parsed.pages else ([parsed.text] if parsed.text else [])
    child_chunks: list[dict] = []
    for pidx, page_text in enumerate(page_texts, start=1):
        if not page_text.strip():
            continue
        for c in rt.chunker.chunk(page_text, doc_id="evaldoc", page_num=pidx):
            if c["chunk_type"] == "child":
                child_chunks.append({**c, "page_num": pidx})

    index_chunks(rt, [
        {"id": c["id"], "content": c["content"],
         "metadata": {"kb_id": KB_ID, "owner_id": OWNER_ID, "doc_id": "evaldoc",
                      "doc_name": os.path.basename(path), "page_num": c["page_num"],
                      "content": c["content"]}}
        for c in child_chunks])
    return child_chunks


def _rank_lists(rt, query: str, top_n: int = 20):
    """分别返回:(hybrid+rerank, hybrid, 干净向量, 干净BM25) 的排名列表(元素含 chunk_id)。"""
    from app.core.retriever import rrf_fuse
    fm = {"kb_id": KB_ID, "owner_id": OWNER_ID}
    qvec = rt.embedding.encode([query])[0]

    v_hits = rt.vector_store.search(qvec, top_k=top_n, filter_meta=fm)
    v_list = [{"chunk_id": h.id, "content": h.metadata.get("content", ""), "score": h.score} for h in v_hits]
    b_list = [{"chunk_id": d["chunk_id"], "content": d.get("content", ""), "score": d.get("score", 0.0)}
              for d in rt.bm25.search(query, top_k=top_n, filter_meta=fm)]

    merged = rrf_fuse(v_list, b_list, k=rt.retriever.rrf_k, top_k=top_n)
    reranked = rt.reranker.rerank(query, merged) if rt.reranker else merged
    return reranked, merged, v_list, b_list



def _gold_ids(chunks: list[dict], g: dict) -> set:
    """含期望事实（声明了页码时还要求页码相符）的 child 块 —— 该问题「应该被捞到」的那些。"""
    want = normalize(g["expect"])
    return {c["id"] for c in chunks
            if want in normalize(c["content"])
            and (g.get("page") is None or c["page_num"] == g["page"])}


def _retrieve_fn(rt):
    """把运行时包成核心要的 retrieve_fn：四种检索方式的排名 + 向量 top-1 相似度。"""

    def retrieve(question: str) -> dict:
        reranked, merged, v_list, b_list = _rank_lists(rt, question)
        qvec = rt.embedding.encode([question])[0]
        top = rt.vector_store.search(qvec, top_k=1,
                                     filter_meta={"kb_id": KB_ID, "owner_id": OWNER_ID})
        return {
            "ranks": {
                "hybrid+rerank": [x["chunk_id"] for x in reranked],
                "hybrid": [x["chunk_id"] for x in merged],
                "vector": [x["chunk_id"] for x in v_list],
                "bm25": [x["chunk_id"] for x in b_list],
            },
            "top1": round(top[0].score, 3) if top else None,
            "top1_id": top[0].id if top else None,
        }

    return retrieve


def main() -> None:
    """离线检索评测。

    **不开降级**（票 39 / #48）：重排不可达就让它抛，这一段照仓库惯例写「未跑」，
    而不是把 RRF 原顺序的数字当成「重排已跑」报出去。

    ⚠️ `get_settings()` 是 lru_cache 单例，而一页报告是**同一个进程**里顺次跑各段的 ——
    改了不还回去，后面跑的段会跟着变严格、结果还依赖段序。所以这里用完就恢复。
    """
    from app.config import get_settings

    settings = get_settings()
    prev = settings.rerank_strict
    settings.rerank_strict = True
    try:
        _main_body()
    finally:
        settings.rerank_strict = prev


def _main_body() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    os.environ.setdefault("PARSER_USE_DOCLING", "false")
    from app.config import get_settings
    from app.core.container import build_runtime

    docs = [d for d in os.environ.get("EVAL_DOCS", "").split(",") if d.strip()] or [DOC]
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    lines: list[str] = [
        "=== RAG 检索质量评估(离线) ===",
        "嵌入: %s | 文档数: %d" % (embedding_label(), len(docs)),
        "黄金集: %s" % GOLDEN,
    ]
    rt = build_runtime()

    chunks_by_doc: dict[str, list[dict]] = {}
    t0 = time.time()
    for path in docs:
        name = os.path.basename(path)
        chunks_by_doc[name] = _index_doc(rt, path)
        lines.append("索引 %s: %d child" % (name, len(chunks_by_doc[name])))
    lines.append("索引耗时 %.1fs" % (time.time() - t0))
    lines.append("")

    def doc_of(g: dict) -> str:
        """条目归属哪份文档：黄金集写了 doc 就用它，否则算在第一份上。"""
        return g.get("doc") or (os.path.basename(docs[0]) if docs else "")

    questions: list[dict] = []
    for g in golden:
        if g.get("negative"):
            questions.append({"question": g["question"], "doc": doc_of(g), "negative": True})
        else:
            questions.append({"question": g["question"], "doc": doc_of(g),
                              "expect": g.get("expect", ""),
                              "gold_ids": _gold_ids(chunks_by_doc.get(doc_of(g), []), g)})

    metrics = run_retrieval_eval(questions, _retrieve_fn(rt), ks=K_LIST,
                                 threshold=get_settings().min_relevance)
    lines.extend(metrics.to_lines())

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        pass


if __name__ == "__main__":
    main()
