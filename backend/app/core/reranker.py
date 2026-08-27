"""Reranker 抽象与实现（重排是检索质量最高 ROI 的一步）。

- FakeReranker: 用查询/文档词重叠给出确定性分数（无需模型），demo/测试可用。
- BgeReranker:  真实 bge-reranker-large（惰性 import transformers，需 GPU）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings

_RELEVANCE_OFFSET = 0.15  # Fake 分数叠加一个常数，保证与检索阈值语义兼容


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
        """输入候选（已含 retrieval score），输出按重排分数排序的列表（新增 'rank_score'）。"""
        ...


class FakeReranker(Reranker):
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
        q = set(str(query).lower().split())
        out = []
        for c in candidates:
            overlap = len(q & set(str(c.get("content", "")).lower().split()))
            base = float(c.get("score") or 0.0)
            c = dict(c)
            c["rank_score"] = round(base + overlap * 0.05 + _RELEVANCE_OFFSET, 4)
            out.append(c)
        return sorted(out, key=lambda x: x["rank_score"], reverse=True)


class BgeReranker(Reranker):
    def __init__(self, model_name: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # 惰性

        s = get_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or s.reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name or s.reranker_model)
        self.model.eval()

    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
        import torch  # 惰性

        pairs = [(query, c.get("content", "")) for c in candidates]
        enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**enc).logits.squeeze(-1).tolist()
        out = []
        for c, sc in zip(candidates, scores):
            c = dict(c)
            c["rank_score"] = float(sc)
            out.append(c)
        return sorted(out, key=lambda x: x["rank_score"], reverse=True)


def get_reranker(provider: str | None = None) -> Reranker:
    provider = provider or get_settings().reranker_provider
    if provider == "bge":
        return BgeReranker()
    return FakeReranker()
