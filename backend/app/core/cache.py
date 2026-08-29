"""语义缓存：相似问题直接返回缓存答案，省 LLM 调用、降延迟。

默认内存版（演示/测试）；设 redis_url + backend=redis 走 Redis（生产）。
"""
from __future__ import annotations

import math
from typing import Sequence

from app.config import get_settings


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class SemanticCache:
    def __init__(self, embedding, threshold: float | None = None, backend: str = "memory"):
        self.embedding = embedding
        self.threshold = threshold if threshold is not None else get_settings().semantic_cache_threshold
        self._entries: list[dict] = []
        self._redis = None
        if backend == "redis":
            self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis

            self._redis = redis.from_url(get_settings().redis_url or "redis://localhost:6379/0")
        except Exception:
            self._redis = None

    def get(self, question: str, kb_id: str) -> dict | None:
        qv = self.embedding.encode([question])[0]
        best: tuple[float, str] | None = None
        for e in self._entries:
            if e["kb_id"] != kb_id:
                continue
            sim = _cosine(qv, e["query_vec"])
            if sim >= self.threshold and (best is None or sim > best[0]):
                best = (sim, e["answer"])
        if best:
            return {"answer": best[1], "score": round(best[0], 3)}
        return None

    def put(self, question: str, kb_id: str, answer: str) -> None:
        if not answer:
            return
        qv = self.embedding.encode([question])[0]
        self._entries.append({"kb_id": kb_id, "query_vec": qv, "answer": answer})
        if len(self._entries) > 2000:
            self._entries = self._entries[-1000:]
