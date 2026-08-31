"""文档入库管线：解析 → 按页分块 → 向量化 → 同时写关系库 + 向量库 + BM25 索引。

reindex_all: 容器重启后内存向量/BM25 索引清空，从数据库的已入库分块重建，
使"重启后仍能检索"，不用重新上传。
"""
from __future__ import annotations

import os

from sqlalchemy.orm import Session

from app.core.container import Runtime
from app.core.source_docs import make_meta
from app.core.vector_store import VectorItem
from app.models.entities import Chunk, Document

# 上传处理进度（0-100，内存态；重启后按 DB status 判断）
_PROGRESS: dict[str, int] = {}


def _set_progress(doc_id: str, val: int) -> None:
    _PROGRESS[doc_id] = val


def get_progress(doc_id: str) -> int:
    """返回处理进度；非处理中（indexed/failed/不存在）返回 100/0。"""
    if doc_id in _PROGRESS:
        return _PROGRESS[doc_id]
    return 100


def _index_units(rt: Runtime, child_units: list[dict], doc: Document) -> None:
    """把子块同时写入向量库 + BM25（含元数据：kb/owner/doc/page/content）。"""
    texts = [c["content"] for c in child_units]
    vectors = rt.embedding.encode(texts)
    items: list[VectorItem] = []
    bm25_entries: list[dict] = []
    for c, vec in zip(child_units, vectors):
        meta = make_meta(doc=doc, content=c["content"], page_num=c["page_num"])
        items.append(VectorItem(id=c["id"], vector=vec, metadata=meta))
        bm25_entries.append({"id": c["id"], "content": c["content"], "metadata": dict(meta)})
    rt.vector_store.add(items)
    rt.bm25.add(bm25_entries)


def process_document(db: Session, rt: Runtime, doc: Document) -> Document:
    try:
        doc.status = "processing"
        db.commit()
        _set_progress(doc.id, 5)
        if not os.path.exists(doc.file_path):
            raise FileNotFoundError(doc.file_path)
        _set_progress(doc.id, 15)

        parsed = rt.parser.parse(doc.file_path, doc.filename)
        _set_progress(doc.id, 35)
        if parsed.metadata.get("error"):
            raise ValueError(f"解析失败: {parsed.metadata['error']}")
        doc.page_count = parsed.page_count
        print(f"[doc] parser={parsed.metadata.get('parser', parsed.metadata.get('kind'))} "
              f"chars={len(parsed.text)} table={'|' in parsed.text}")

        page_texts = parsed.pages if parsed.pages else ([parsed.text] if parsed.text else [])
        all_chunks: list[dict] = []
        for pidx, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            all_chunks.extend(rt.chunker.chunk(page_text, doc_id=doc.id, page_num=pidx))

        child_units = [c for c in all_chunks if c["chunk_type"] == "child"]
        _set_progress(doc.id, 55)
        print(f"[doc] {doc.filename}: pages={doc.page_count} all_chunks={len(all_chunks)} child={len(child_units)}")

        # 写关系库
        db.add_all([
            Chunk(id=c["id"], parent_id=c["parent_id"], doc_id=doc.id, kb_id=doc.kb_id,
                  owner_id=doc.owner_id, content=c["content"], parent_content=c["parent_content"],
                  page_num=c["page_num"], chunk_type=c["chunk_type"])
            for c in all_chunks
        ])
        db.commit()

        import time as _t
        t0 = _t.time()
        _set_progress(doc.id, 70)
        _index_units(rt, child_units, doc)
        print(f"[doc] {doc.filename}: embed+index {len(child_units)} chunks in {_t.time()-t0:.1f}s")
        _set_progress(doc.id, 95)
        doc.chunk_count = len(child_units)
        doc.status = "indexed"
        db.commit()
        _set_progress(doc.id, 100)
    except Exception as e:  # noqa
        print(f"[doc] {doc.filename} FAILED: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        _set_progress(doc.id, 0)
        try:
            d2 = db.get(Document, doc.id)
            if d2:
                d2.status = "failed"
                d2.error = str(e)
                db.commit()
        except Exception:
            pass
    return doc


def reindex_all(db: Session, rt: Runtime) -> int:
    """从数据库已入库的 child 分块重建内存向量库 + BM25 索引。返回重建的分块数。"""
    rows = (db.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.chunk_type == "child", Document.status == "indexed").all())
    if not rows:
        return 0
    chunks = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    texts = [c.content for c in chunks]
    vectors = rt.embedding.encode(texts)
    items: list[VectorItem] = []
    bm25_entries: list[dict] = []
    for c, d, vec in zip(chunks, docs, vectors):
        meta = {"kb_id": c.kb_id, "owner_id": c.owner_id, "doc_id": c.doc_id,
                "doc_name": d.filename, "page_num": c.page_num, "content": c.content}
        items.append(VectorItem(id=c.id, vector=vec, metadata=meta))
        bm25_entries.append({"id": c.id, "content": c.content, "metadata": dict(meta)})
    rt.vector_store.add(items)
    rt.bm25.add(bm25_entries)
    return len(chunks)


import threading

_processing_lock = threading.Lock()  # 串行处理：避免并发上传时 bm25/向量库/GPU 争用


def process_document_background(doc_id: str) -> None:
    """后台线程：用独立 session + 全局 runtime 单例处理一个文档。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal

    with _processing_lock:
        db: Session = SessionLocal()
        try:
            doc = db.get(Document, doc_id)
            if doc:
                process_document(db, get_runtime(), doc)
        finally:
            db.close()


def launch_processing(doc_id: str) -> None:
    """上传接口调用：立刻返回，解析/嵌入在后台线程执行。"""
    threading.Thread(target=process_document_background, args=(doc_id,), daemon=True).start()
