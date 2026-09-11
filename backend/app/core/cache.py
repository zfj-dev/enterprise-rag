"""语义缓存：相似问题直接返回缓存答案，省 LLM 调用、降延迟。

默认内存版（演示/测试）；设 redis_url + backend=redis 走 Redis（生产）。
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.core.similarity import cosine

logger = logging.getLogger(__name__)

_REDIS_KEY = "rag:semcache"


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
        except Exception as e:
            logger.warning("Redis 初始化失败,回退内存缓存: %s", e)
            self._redis = None

    def _load_entries(self) -> list[dict]:
        if self._redis:
            try:
                import json

                raw = self._redis.lrange(_REDIS_KEY, 0, -1)
                return [json.loads(x) for x in raw]
            except Exception:
                pass
        return self._entries

    def _save_entries(self, entries: list[dict]) -> None:
        if self._redis:
            try:
                import json
                import redis as _r

                pipe = self._redis.pipeline()
                pipe.delete(_REDIS_KEY)
                for e in entries[-2000:]:
                    pipe.rpush(_REDIS_KEY, json.dumps(e, ensure_ascii=False))
                pipe.execute()
            except Exception:
                self._entries = entries
        else:
            self._entries = entries

    def get(self, question: str, kb_id: str) -> dict | None:
        qv = self.embedding.encode([question])[0]
        best: tuple[float, str] | None = None
        for e in self._load_entries():
            if e["kb_id"] != kb_id:
                continue
            sim = cosine(qv, e["query_vec"])
            if sim >= self.threshold and (best is None or sim > best[0]):
                best = (sim, e["answer"])
        if best:
            return {"answer": best[1], "score": round(best[0], 3)}
        return None

    def put(self, question: str, kb_id: str, answer: str) -> None:
        if not answer:
            return
        qv = self.embedding.encode([question])[0]
        entries = self._load_entries()
        entries.append({"kb_id": kb_id, "query_vec": qv, "answer": answer})
        self._save_entries(entries[-2000:])
