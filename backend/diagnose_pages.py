"""诊断：docling 按页导出的 markdown 是否真的分页，chunk 的 page_num 是否正确。
用法: .venv\\Scripts\\python.exe diagnose_pages.py
"""
from __future__ import annotations
import os, sys

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

def main():
    from app.core.parser import ParserRouter
    from app.core.chunker import ParentChildChunker
    pdf = os.path.join(BACKEND, "paper.pdf")
    parsed = ParserRouter().parse(pdf, "paper.pdf")
    print(f"parser={parsed.metadata.get('parser')}  page_count={parsed.page_count}  len(pages)={len(parsed.pages)}")

    # 每页概览（只看前 3 页 + 含关键内容的页）
    for i, pg in enumerate(parsed.pages, 1):
        if i <= 3 or "RTX 3060" in pg or "表 3 . 1" in pg:
            print(f"  page {i}: chars={len(pg)}  head={pg[:45]!r}  has_rtx={'RTX 3060' in pg}")

    # RTX 3060 / 表3.1 在哪个 page
    for i, pg in enumerate(parsed.pages, 1):
        if "RTX 3060" in pg:
            print(f"  >>> RTX 3060 出现在 page {i}（引用应显示第{i}页）")
            break

    # 模拟 chunk，看 page_num
    chunks = []
    for pidx, pg in enumerate(parsed.pages, 1):
        if not pg.strip():
            continue
        chunks.extend(ParentChildChunker().chunk(pg, doc_id="diag", page_num=pidx))
    childs = [c for c in chunks if c["chunk_type"] == "child"]
    rtx = [c for c in childs if "RTX 3060" in c["content"]]
    if rtx:
        print(f"  >>> 含 RTX 3060 的 chunk page_num = {rtx[0]['page_num']}")
    print(f"  总计 child chunks = {len(childs)}")

if __name__ == "__main__":
    main()
