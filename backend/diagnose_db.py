"""直查服务器 sqlite 数据库：dump 表3.2/3.3 相关 chunk（空白容错），验证内容。
用法: .venv\\Scripts\\python.exe diagnose_db.py
"""
from __future__ import annotations
import os, re, sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("DATABASE_URL", "sqlite:///./rag.db")

def main():
    from sqlalchemy import create_engine
    from app.models.entities import Chunk, Document
    from sqlalchemy.orm import Session

    engine = create_engine(os.environ["DATABASE_URL"])
    with Session(engine) as db:
        rows = (db.query(Chunk, Document.filename)
                .join(Document, Chunk.doc_id == Document.id)
                .filter(Chunk.chunk_type == "child")
                .all())
        print(f"child chunks 总数: {len(rows)}")
        for tid in ["3.2", "3.3"]:
            key = f"表{tid}"
            hits = []
            for r, fn in rows:
                compact = re.sub(r"\s+", "", r.content or "")
                if key in compact:
                    hits.append((r, fn))
            print(f"\n=== 含紧凑 '{key}' 的 chunk（{len(hits)} 个）===")
            for r, fn in hits[:5]:
                print(f"[{r.id}] page={r.page_num} doc={fn}")
                print("   " + (r.content or "")[:350].replace("\n", " / "))
        # 额外：打印 page 30 和 33 的所有 child chunk 开头，确认表归属
        print("\n=== page 29/30/33 的所有 child chunk ===")
        for pno in [29, 30, 33]:
            pchunks = [r for r, fn in rows if r.page_num == pno]
            print(f"\n-- page {pno} ({len(pchunks)} chunks) --")
            for r in pchunks[:6]:
                print(f"  [{r.id}] head={(r.content or '')[:70]!r}")

if __name__ == "__main__":
    main()
