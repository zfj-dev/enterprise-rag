"""诊断：各表 chunk 内容 + 表N查询的检索排名，定位表3.2/3.3 检索问题。
用法: .venv\\Scripts\\python.exe diagnose_tables.py   报告 logs/table-retrieval-diagnose.log
"""
from __future__ import annotations
import os, re, sys, time

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
for k, v in {
    "HF_HUB_OFFLINE": "0", "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_SYMLINKS": "1", "HF_HUB_DISABLE_XET": "1",
    "USE_REAL": "true", "EMBEDDING_PROVIDER": "bge", "RERANKER_PROVIDER": "bge",
    "EMBEDDING_DEVICE": "cuda", "RERANKER_DEVICE": "cuda", "VECTOR_STORE": "inmemory",
}.items():
    os.environ.setdefault(k, v)

REPORT = os.path.join(BACKEND, "logs", "table-retrieval-diagnose.log")
_lines = []
def log(s=""):
    _lines.append(str(s)); print(s, flush=True)

def main():
    pdf = os.path.join(BACKEND, "paper.pdf")
    from app.core.parser import ParserRouter
    from app.core.chunker import ParentChildChunker
    from app.core.embedding import get_embedding
    from app.core.vector_store import InMemoryVectorStore, VectorItem
    from app.core.bm25 import InMemoryBm25
    from app.core.reranker import get_reranker
    from app.core.retriever import HybridRetriever

    parsed = ParserRouter().parse(pdf, "paper.pdf")
    log(f"[1] parser={parsed.metadata.get('parser')} pages={parsed.page_count} len(pages)={len(parsed.pages)}")

    chunks = []
    for pidx, pg in enumerate(parsed.pages, 1):
        if not pg.strip(): continue
        chunks.extend(ParentChildChunker().chunk(pg, doc_id="diag", page_num=pidx))
    childs = [c for c in chunks if c["chunk_type"] == "child"]
    log(f"[2] child chunks={len(childs)}")

    # 各表：含 caption 的 chunk
    for tid in ["3.1", "3.2", "3.3", "3.4", "4.1", "4.2"]:
        cap_re = re.compile(rf"表\s*{re.escape(tid)}")
        caps = [c for c in childs if cap_re.search(c["content"])]
        log(f"\n=== 表{tid}: {len(caps)} 个含'表{tid}'的 chunk ===")
        for c in caps[:3]:
            log(f"  [{c['id']}] page={c['page_num']}")
            log("    " + c["content"][:140].replace("\n", " / "))

    # 建索引 + 检索
    t0 = time.time()
    embed = get_embedding()
    vecs = embed.encode([c["content"] for c in childs])
    vs = InMemoryVectorStore(); bm = InMemoryBm25()
    vs.add([VectorItem(id=c["id"], vector=v, metadata={"content": c["content"], "page_num": c["page_num"]}) for c, v in zip(childs, vecs)])
    bm.add([{"id": c["id"], "content": c["content"], "metadata": {"content": c["content"]}} for c in childs])
    log(f"\n[3] 索引 {len(childs)} 块: {time.time()-t0:.0f}s")
    rt = HybridRetriever(vs, bm, embed, reranker=get_reranker())

    for q in ["表3.1的内容是什么", "表3.2的内容是什么", "表3.3的内容是什么", "表3.4的内容是什么"]:
        hits = rt.retrieve(q, top_k=3)
        log(f"\n=== 检索: {q} ===")
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            log(f"  {i}. [{h.get('chunk_id')}] page={md.get('page_num')}")
            log("     " + (h.get("content", "") or "")[:110].replace("\n", " / "))

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[doc] 报告已写入 {REPORT}")

if __name__ == "__main__":
    main()
