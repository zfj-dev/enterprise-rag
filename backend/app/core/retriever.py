"""混合检索引擎：向量(pgvector/内存) + BM25 + RRF 融合 + （可选）重排。"""
from __future__ import annotations

from typing import Sequence

from app.config import get_settings
from app.core.bm25 import InMemoryBm25
from app.core.embedding import EmbeddingModel
from app.core.reranker import Reranker
from app.core.vector_store import VectorStore


def rrf_fuse(
    vector_hits: Sequence[dict],
    bm25_hits: Sequence[dict],
    k: int = 60,
    top_k: int = 20,
    keep_attr: str = "content",
) -> list[dict]:
    """Reciprocal Rank Fusion：按候选在两个列表中的名次打分，融合去重。返回值按融合分排序。"""
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for hits in (vector_hits, bm25_hits):
        for rank, hit in enumerate(hits, start=1):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best:
                best[cid] = hit
    merged = []
    for cid, sc in scores.items():
        item = dict(best[cid])
        item["rrf_score"] = round(sc, 5)
        item["rank_score"] = item.get("rank_score", round(sc, 5))
        merged.append(item)
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25: InMemoryBm25,
        embedding: EmbeddingModel,
        reranker: Reranker | None = None,
        retrieval_top_k: int | None = None,
        rrf_k: int | None = None,
    ):
        s = get_settings()
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedding = embedding
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k or s.retrieval_top_k
        self.rrf_k = rrf_k or s.rrf_k

    def retrieve(
        self,
        query: str,
        kb_id: str | None = None,
        owner_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        s = get_settings()
        filter_meta: dict = {}
        if kb_id is not None:
            filter_meta["kb_id"] = kb_id
        if owner_id is not None:
            filter_meta["owner_id"] = owner_id

        qvec = self.embedding.encode([query])[0]
        v_hits = []
        for hit in self.vector_store.search(qvec, top_k=self.retrieval_top_k, filter_meta=filter_meta or None):
            v_hits.append({"chunk_id": hit.id, "content": hit.metadata.get("content", ""),
                           "score": hit.score, "metadata": hit.metadata})

        b_hits = self.bm25.search(query, top_k=self.retrieval_top_k, filter_meta=filter_meta or None)

        merged = rrf_fuse(v_hits, b_hits, k=self.rrf_k, top_k=top_k or s.rerank_top_k * 4)

        if self.reranker:
            merged = self.reranker.rerank(query, merged)

        final_top = top_k or s.rerank_top_k
        return merged[:final_top]
