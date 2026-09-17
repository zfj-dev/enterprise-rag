"""BM25 关键词检索（InMemory 实现，用 rank_bm25；支持按元数据过滤）。"""
from __future__ import annotations

import re
import threading

from dataclasses import dataclass, field
from typing import Any, Sequence

from rank_bm25 import BM25Okapi


@dataclass
class _Doc:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryBm25:
    """全局语料的 BM25 索引；search 后按元数据过滤再返回 Top-K。"""

    def __init__(self, tokenizer=None):
        self._docs: dict[str, _Doc] = {}
        self._bm25: BM25Okapi | None = None
        self._corpus: list[str] = []
        # `add` 是**先写 _docs 再 _rebuild**：那一瞬间 `_corpus` 与 `_docs` 长度都不一致，
        # 并发 search 会拿错位（甚至越界）的分数，边遍历边改还会 `RuntimeError`。
        # 读写共用一把锁，顺带把「打分与文档对不上」一起关掉。
        self._lock = threading.RLock()

    def _tok(self, text: str) -> list[str]:
        """中文感知分词：去空白后，ASCII 词整词保留，中文按字符双字组。

        修复中文无空格时 BM25 整句挤成 1 个 token 的问题（如"表3.1""比亚迪"等标识/关键词能命中）。
        """
        t = re.sub(r"\s+", "", str(text).lower())
        if not t:
            return []
        words = re.findall(r"[a-z0-9]+", t)
        rest = re.sub(r"[a-z0-9]+", "", t)
        grams = list(words)
        if len(rest) == 1:
            grams.append(rest)
        else:
            for i in range(len(rest) - 1):
                grams.append(rest[i:i + 2])
        return grams

    def add(self, docs: Sequence[Any]) -> None:
        with self._lock:
            for d in docs:
                if isinstance(d, dict):
                    self._docs[d["id"]] = _Doc(id=d["id"], content=d.get("content", ""),
                                               metadata=d.get("metadata", {}))
                else:
                    self._docs[d.id] = d
            self._rebuild()

    def _rebuild(self) -> None:
        self._corpus = [d.content for d in self._docs.values()]
        self._bm25 = BM25Okapi([self._tok(c) for c in self._corpus]) if self._corpus else None

    def search(self, query: str, top_k: int = 10, filter_meta: dict | None = None) -> list[dict]:
        with self._lock:
            if not self._bm25 or not self._docs:
                return []
            scores = self._bm25.get_scores(self._tok(query))
            hits = []
            # 分数按 `_corpus` 的下标对齐 —— 要按下标取，不能用 `list(...).index(doc_id)`
            # （那是 O(n²)，而且在并发改动下会把分数贴到别人头上）。
            for i, (doc_id, doc) in enumerate(self._docs.items()):
                if filter_meta and any(doc.metadata.get(k) != v for k, v in filter_meta.items()):
                    continue
                hits.append({"chunk_id": doc_id, "content": doc.content,
                             "score": float(scores[i]),
                             "metadata": dict(doc.metadata)})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]

    def remove_by(self, doc_id: str | None = None, kb_id: str | None = None) -> None:
        with self._lock:
            for key in list(self._docs.keys()):
                m = self._docs[key].metadata
                if (doc_id is None or m.get("doc_id") == doc_id) and (kb_id is None or m.get("kb_id") == kb_id):
                    self._docs.pop(key, None)
            self._rebuild()
