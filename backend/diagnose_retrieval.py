"""检索诊断：当前代码下 解析->分块->建索引->查询"表3.1的内容是什么" 命中什么。
用法: scripts\\diagnose_retrieval.ps1   报告 logs/retrieval-diagnose.log
"""
from __future__ import annotations
import os, sys, time

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

for k, v in {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_OFFLINE": "0",
    "HF_HUB_DISABLE_SYMLINKS": "1",
    "HF_HUB_DISABLE_XET": "1",
    "USE_REAL": "true",
    "EMBEDDING_PROVIDER": "bge",
    "RERANKER_PROVIDER": "bge",
    "EMBEDDING_DEVICE": "cuda",
    "RERANKER_DEVICE": "cuda",
    "VECTOR_STORE": "inmemory",
}.items():
    os.environ.setdefault(k, v)

REPORT = os.path.join(BACKEND, "logs", "retrieval-diagnose.log")
_lines: list[str] = []
def log(s: str = "") -> None:
    _lines.append(str(s)); print(s, flush=True)

def main() -> int:
    pdf = os.path.join(BACKEND, "paper.pdf")
    t0 = time.time()
    from app.core.parser import ParserRouter
    parsed = ParserRouter().parse(pdf, "paper.pdf")
    log(f"[1] 解析: parser={parsed.metadata.get('parser')} chars={len(parsed.text)} ({time.time()-t0:.0f}s)")

    from app.core.chunker import ParentChildChunker
    units = ParentChildChunker().chunk(parsed.text, doc_id="diag", page_num=1)
    children = [c for c in units if c["chunk_type"] == "child"]
    log(f"[2] 分块: child={len(children)}")

    tbl = [c for c in children if "RTX 3060" in c["content"]]
    if tbl:
        log(f"[3] 表格 chunk: {tbl[0]['id']}  含标题'表 3 . 1'={'表 3 . 1' in tbl[0]['content']}")
        log("    内容开头: " + tbl[0]["content"][:140].replace("\n", " / "))
    else:
        log("[3] !! 未找到含 RTX 3060 的表格 chunk")

    from app.core.embedding import get_embedding
    from app.core.vector_store import InMemoryVectorStore, VectorItem
    from app.core.bm25 import InMemoryBm25
    from app.core.reranker import get_reranker
    from app.core.retriever import HybridRetriever

    t0 = time.time()
    embed = get_embedding()
    vecs = embed.encode([c["content"] for c in children])
    vs = InMemoryVectorStore()
    bm = InMemoryBm25()
    vs.add([VectorItem(id=c["id"], vector=v, metadata={"content": c["content"]}) for c, v in zip(children, vecs)])
    bm.add([{"id": c["id"], "content": c["content"], "metadata": {}} for c in children])
    log(f"[4] 编码+入库 {len(children)} 块: {time.time()-t0:.0f}s")

    retriever = HybridRetriever(vs, bm, embed, reranker=get_reranker())
    q = "表3.1的内容是什么"
    hits = retriever.retrieve(q, top_k=8)
    log(f"\n[5] 查询: {q}")
    for i, h in enumerate(hits, 1):
        content = h.get("content", "")
        tag = "【表格】" if "RTX 3060" in content else ""
        log(f"  {i}. [{h.get('chunk_id')}] rrf={h.get('rrf_score')} {tag}")
        log("     " + content[:100].replace("\n", " / "))
    rank = next((i for i, h in enumerate(hits, 1) if "RTX 3060" in h.get("content", "")), None)
    log(f"\n表格 chunk 命中排名: #{rank if rank else '不在 top-8'}")

    qv = embed.encode([q])[0]
    vhits = vs.search(qv, top_k=10)
    log("\n[对照] 纯向量 top-10:")
    for i, h in enumerate(vhits, 1):
        tag = "【表格】" if "RTX 3060" in h.metadata.get("content", "") else ""
        log(f"  {i}. [{h.id}] score={h.score:.4f} {tag} {h.metadata.get('content','')[:60].replace(chr(10),' / ')}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[doc] 报告已写入 {REPORT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
