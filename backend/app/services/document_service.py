"""文档入库管线：解析 → 分块 → 向量化 → 同时写关系库 + 向量库 + BM25 索引。"""
from __future__ import annotations

import os
from sqlalchemy.orm import Session

from app.core.container import Runtime
from app.core.source_docs import make_meta
from app.core.vector_store import VectorItem
from app.models.entities import Chunk, Document


def process_document(db: Session, rt: Runtime, doc: Document) -> Document:
    """解析并索引一个文档；就地更新其状态/分块数。"""
    try:
        doc.status = "processing"
        db.commit()
        if not os.path.exists(doc.file_path):
            raise FileNotFoundError(doc.file_path)

        parsed = rt.parser.parse(doc.file_path, doc.filename)
        if parsed.metadata.get("error"):
            raise ValueError(f"解析失败: {parsed.metadata['error']}")
        text = parsed.text
        doc.page_count = parsed.pages

        # 分块
        chunk_units = rt.chunker.chunk(text, doc_id=doc.id, page_num=0)

        # 只对"子块"建索引（检索用）；父块入库用于生成上下文
        child_units = [c for c in chunk_units if c["chunk_type"] == "child"]

        # 写入关系库
        db_rows: list[Chunk] = []
        for c in chunk_units:
            row = Chunk(
                id=c["id"], parent_id=c["parent_id"], doc_id=doc.id,
                kb_id=doc.kb_id, owner_id=doc.owner_id,
                content=c["content"], parent_content=c["parent_content"],
                page_num=c["page_num"], chunk_type=c["chunk_type"],
            )
            db_rows.append(row)
        db.add_all(db_rows)
        db.commit()

        # 向量化 + 索引（子块）
        texts = [c["content"] for c in child_units]
        vectors = rt.embedding.encode(texts)
        items: list[VectorItem] = []
        for c, vec in zip(child_units, vectors):
            meta = make_meta(doc=doc, content=c["content"], page_num=c["page_num"])
            items.append(VectorItem(id=c["id"], vector=vec, metadata=meta))
        rt.vector_store.add(items)

        # Bm25 索引（含内容+元数据）
        rm_meta_entries = []
        for c in child_units:
            rm_meta_entries.append({"id": c["id"], "content": c["content"],
                                    "metadata": make_meta(doc=doc, content=c["content"], page_num=c["page_num"])})
        rt.bm25.add([{"id": e["id"], "content": e["content"], "metadata": e["metadata"]} for e in rm_meta_entries])

        doc.chunk_count = len(child_units)
        doc.status = "indexed"
        db.commit()
    except Exception as e:  # noqa
        doc.status = "failed"
        doc.error = str(e)
        db.commit()
    return doc
