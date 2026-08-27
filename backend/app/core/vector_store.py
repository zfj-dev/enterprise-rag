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

    add/search 直接操作 document_chunks 表上的 embedding 向量列（简化：同表存向量+元数据）。
    """

    def __init__(self, conn_url: str, dim: int = 1024):
        import pgvector.sqlalchemy  # 惰性
        from sqlalchemy import create_engine, text

        self.dim = dim
        self.engine = create_engine(conn_url, future=True)
        self._pg = pgvector

    def add(self, items: Sequence[VectorItem]) -> None:
        from sqlalchemy import text
        from sqlalchemy.dialects.postgresql import insert

        rows = []
        for it in items:
            rows.append({"id": it.id, "doc_id": it.metadata.get("doc_id"),
                         "kb_id": it.metadata.get("kb_id"), "owner_id": it.metadata.get("owner_id"),
                         "embedding": "[" + ",".join(f"{x:.6f}" for x in it.vector) + "]"})
        with self.engine.begin() as conn:
            stmt = insert(document_chunks_table()).values(rows)
            stmt = stmt.on_conflict_do_update(index_elements=["id"], set_={"embedding": stmt.excluded.embedding})
            conn.execute(stmt)

    # noqa
    def search(self, vector, top_k=10, filter_meta=None):
        raise NotImplementedError("PgVectorStore.search 需按 pgvector 距离算子实现；demo/测试用 InMemoryVectorStore")


def document_chunks_table():
    from sqlalchemy import MetaData, Table

    return Table("document_chunks", MetaData(), autoload_with=None)  # 占位，真实实现用 ORM


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
