"""Reranker 抽象与实现（重排是检索质量最高 ROI 的一步）。

- FakeReranker: 词重叠分数（演示/测试）。
- BgeReranker:  真实 bge-reranker-large（本地 GPU，需 torch + transformers；宿主跑）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings

_RELEVANCE_OFFSET = 0.15


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
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
    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        s = get_settings()
        dev = device or s.reranker_device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self._torch = torch
        self.device = dev
        print(f"[bge] reranker device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or s.reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name or s.reranker_model)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
        pairs = [(query, c.get("content", "")) for c in candidates]
        enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.no_grad():
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
