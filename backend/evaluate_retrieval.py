"""离线检索质量评估：量化"检索层"是否把含答案的 chunk 捞进 top-k。

与 evaluate.py(端到端:fact_hit/grounded/page)互补。本脚本不依赖运行中的服务、不需要 LLM：
直接 build_runtime() 解析→分块→索引 paper.pdf,然后用混合/向量/BM25/重排四种检索方式,
对每个问题算 hit-rate@k / recall@k / MRR;负样本算 top-1 相似度(看是否该被拒答)。

用法:  cd backend && python evaluate_retrieval.py
报告:  logs/retrieval-eval-report.log
环境:  EMBEDDING_PROVIDER=bge 时才是真质量(FakeEmbedding 仅演示管线)。
"""
from __future__ import annotations

import json
import os
import re
import time

BACKEND = os.path.dirname(os.path.abspath(__file__))

REPORT = os.path.join(BACKEND, "logs", "retrieval-eval-report.log")
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_retrieval.json"))
KB_ID, OWNER_ID = "eval_kb", "eval_owner"
K_LIST = (3, 5, 10)  # 评估 hit-rate/recall/MRR 的 top-k


def _norm(s) -> str:
    return re.sub(r"\s+", "", s or "").lower()


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

    texts = [c["content"] for c in child_chunks]
    vectors = rt.embedding.encode(texts)
    items, bm25_entries = [], []
    for c, vec in zip(child_chunks, vectors):
        meta = {"kb_id": KB_ID, "owner_id": OWNER_ID, "doc_id": "evaldoc",
                "doc_name": os.path.basename(path), "page_num": c["page_num"], "content": c["content"]}
        items.append({"id": c["id"], "vector": vec, "metadata": meta})  # 借 VectorItem-like dict
        bm25_entries.append({"id": c["id"], "content": c["content"], "metadata": dict(meta)})
    from app.core.vector_store import VectorItem
    rt.vector_store.add([VectorItem(**i) for i in items])
    rt.bm25.add(bm25_entries)
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


def _metrics(ranked: list[dict], gold: set[str], k: int):
    """hit@k: 任一 gold 在前 k; recall@k: 找到的 gold 占 gold 总数; mrr: 首个 gold 的名次倒数。"""
    ids = [x["chunk_id"] for x in ranked[:k]]
    hit = 1.0 if any(g in ids for g in gold) else 0.0
    recall = len(gold & set(ids)) / max(1, len(gold))
    mrr = 0.0
    for i, cid in enumerate(ids, start=1):
        if cid in gold:
            mrr = 1.0 / i
            break
    return hit, recall, mrr


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    # 离线检索评估只用页码文本即可,走 PyMuPDF,避开 docling 模型下载(生产/端到端仍用 docling)
    os.environ.setdefault("PARSER_USE_DOCLING", "false")
    lines: list[str] = ["=== RAG 检索质量评估(离线) ==="]
    lines.append(f"文档: {DOC} | 嵌入: {os.environ.get('EMBEDDING_PROVIDER','fake')} | 检索/重排配置见 config")
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    from app.core.container import build_runtime
    rt = build_runtime()

    t0 = time.time()
    chunks = _index_doc(rt, DOC)
    lines.append(f"解析→分块→索引 {len(chunks)} 个 child chunk,耗时 {time.time()-t0:.1f}s")

    fact_questions = [g for g in golden if not g.get("negative")]
    neg_questions = [g for g in golden if g.get("negative")]

    # 每种方法的指标累加器
    acc = {label: {"hit": {k: [0.0] * len(fact_questions) for k in K_LIST},
                   "recall": {k: [0.0] * len(fact_questions) for k in K_LIST},
                   "mrr": [0.0] * len(fact_questions)} for label in ("hybrid+rerank", "hybrid", "vector", "bm25")}

    for gi, g in enumerate(fact_questions):
        q, exp = g["question"], g["expect"]
        gold = {c["id"] for c in chunks if _norm(exp) in _norm(c["content"])
                and (g.get("page") is None or c["page_num"] == g["page"])}
        if not gold:
            lines.append(f"[GOLD缺失] Q:{q} 期望:{exp} —— 未在任一 child chunk 命中,请检查分块/期望值")
            continue
        reranked, merged, v_list, b_list = _rank_lists(rt, q)
        for label, ranked in (("hybrid+rerank", reranked), ("hybrid", merged), ("vector", v_list), ("bm25", b_list)):
            for k in K_LIST:
                h, r, m = _metrics(ranked, gold, k)
                acc[label]["hit"][k][gi] = h
                acc[label]["recall"][k][gi] = r
                acc[label]["mrr"][gi] = m
        hit3 = sum(acc[x]["hit"][3][gi] for x in acc) / len(acc)
        lines.append(f"[{'OK' if hit3 >= 0.5 else '..'}] Q:{q} | 期望:{exp} | gold={len(gold)} |"
                     f" MRR(hybrid+rerank)={acc['hybrid+rerank']['mrr'][gi]:.2f}")

    lines.append("\n=== 汇总(各问题平均) ===")
    for label in acc:
        a = acc[label]
        sweep = [f"@k{k}: hit={sum(a['hit'][k])/len(fact_questions):.2f} rec={sum(a['recall'][k])/len(fact_questions):.2f}" for k in K_LIST]
        lines.append(f"{label:>12}  MRR={sum(a['mrr'])/len(fact_questions):.3f}  " + "  ".join(sweep))

    lines.append("\n=== 负样本(向量top-1相似度应低于 min_relevance → 该被拒答) ===")
    from app.config import get_settings
    min_rel = get_settings().min_relevance
    for g in neg_questions:
        qvec = rt.embedding.encode([g["question"]])[0]
        top = rt.vector_store.search(qvec, top_k=1, filter_meta={"kb_id": KB_ID, "owner_id": OWNER_ID})
        top_score = round(top[0].score, 3) if top else None
        top_cid = top[0].id if top else None
        flag = "低(可拒)" if (top_score or 0) < min_rel else "⚠高(需端到端验证)"
        lines.append(f"[{flag}] Q:{g['question']} | 向量top-1相似度={top_score} chunk={top_cid} (min_relevance={min_rel})")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)
    try:
        print("\n".join(lines))          # 控制台可能是 GBK,遇'⚠'等会失败;报告文件已写入,异常仅影响打印
    except UnicodeEncodeError:
        pass


if __name__ == "__main__":
    main()
