"""向量存储适配层（可按接口切 pgvector / Milvus / Qdrant）。

当前实现：
- InMemoryVectorStore: 纯内存余弦检索（demo/测试，无需数据库）。
- PgVectorStore:       pgvector（真实，惰性 import，需 postgres + pgvector 扩展）。
"""
from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class VectorItem:
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def add(self, items: Sequence[VectorItem]) -> None: ...

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 10, filter_meta: dict | None = None) -> list[SearchHit]: ...

    @abstractmethod
    def delete_by(self, doc_id: str | None = None, kb_id: str | None = None) -> None: ...


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._data: dict[str, tuple[list[float], dict]] = {}

    def add(self, items: Sequence[VectorItem]) -> None:
        for it in items:
            self._data[it.id] = (list(it.vector), dict(it.metadata))

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def _match(self, meta: dict, flt: dict | None) -> bool:
        if not flt:
            return True
        for k, v in flt.items():
            if meta.get(k) != v and (v is not None and meta.get(k) != v):
                return False
        return True

    def search(self, vector: Sequence[float], top_k: int = 10, filter_meta: dict | None = None) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for cid, (vec, meta) in self._data.items():
            if not self._match(meta, filter_meta):
                continue
            hits.append(SearchHit(id=cid, score=self._cosine(vector, vec), metadata=meta))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete_by(self, doc_id: str | None = None, kb_id: str | None = None) -> None:
        keys = [k for k, (_, m) in self._data.items()
                if (doc_id is None or m.get("doc_id") == doc_id)
                and (kb_id is None or m.get("kb_id") == kb_id)]
        for k in keys:
            self._data.pop(k, None)


class PgVectorStore(VectorStore):
    """pgvector 实现（真实）。惰性 import；需要 database_url 指向 postgres + 启用 vector 扩展。

    embedding 与文本共用 document_chunks 表：本类在 __init__ 用幂等 DDL 补 embedding 列
    （共享 ORM Chunk 模型不含该列,sqlite 模式不受影响）。add 只在分块行已由 process_document
    落库后 upsert,因此不会触发其他 NOT NULL 列缺失。
    """

    def __init__(self, conn_url: str, dim: int = 1024):
        from sqlalchemy import create_engine, text

        self.dim = dim
        self.engine = create_engine(conn_url, future=True)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector({dim})"))

    @staticmethod
    def _vec_literal(vec: Sequence[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    def add(self, items: Sequence[VectorItem]) -> None:
        from sqlalchemy import text

        rows = [
            {"id": it.id, "doc_id": it.metadata.get("doc_id"), "kb_id": it.metadata.get("kb_id"),
             "owner_id": it.metadata.get("owner_id"), "embedding": self._vec_literal(it.vector)}
            for it in items
        ]
        with self.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO document_chunks (id, doc_id, kb_id, owner_id, embedding) "
                "VALUES (:id, :doc_id, :kb_id, :owner_id, :embedding) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ), rows)

    def search(self, vector, top_k=10, filter_meta=None):
        from sqlalchemy import text

        filter_meta = filter_meta or {}
        qv = self._vec_literal(vector)
        sql = text(
            "SELECT c.id, c.content, c.page_num, c.doc_id, c.kb_id, c.owner_id, d.filename AS doc_name, "
            "1 - (c.embedding <=> :qv) AS score "
            "FROM document_chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE c.chunk_type = 'child' AND c.embedding IS NOT NULL "
            "AND (:kb_id IS NULL OR c.kb_id = :kb_id) "
            "AND (:owner_id IS NULL OR c.owner_id = :owner_id) "
            "ORDER BY c.embedding <=> :qv "
            "LIMIT :top_k"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"qv": qv, "kb_id": filter_meta.get("kb_id"),
                                      "owner_id": filter_meta.get("owner_id"), "top_k": top_k}).fetchall()
        return [
            SearchHit(id=r.id, score=float(r.score), metadata={
                "kb_id": r.kb_id, "owner_id": r.owner_id, "doc_id": r.doc_id,
                "doc_name": r.doc_name, "page_num": r.page_num, "content": r.content,
            })
            for r in rows
        ]

    def delete_by(self, doc_id: str | None = None, kb_id: str | None = None) -> None:
        from sqlalchemy import text

        conds, params = [], {}
        if doc_id is not None:
            conds.append("doc_id = :doc_id")
            params["doc_id"] = doc_id
        if kb_id is not None:
            conds.append("kb_id = :kb_id")
            params["kb_id"] = kb_id
        if not conds:
            return
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM document_chunks WHERE " + " AND ".join(conds)), params)


def get_vector_store(
    backend: str | None = None,
    conn_url: str | None = None,
    dim: int | None = None,
) -> VectorStore:
    from app.config import get_settings

    s = get_settings()
    backend = backend or s.vector_store
    if backend == "pgvector":
        return PgVectorStore(conn_url or s.database_url, dim or s.embedding_dim)
    return InMemoryVectorStore()
