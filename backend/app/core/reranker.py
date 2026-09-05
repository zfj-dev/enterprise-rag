"""Reranker 抽象与实现（重排是检索质量最高 ROI 的一步）。

- FakeReranker: 词重叠分数（演示/测试）。
- BgeReranker:  真实 bge-reranker-large（本地 GPU，需 torch + transformers；宿主跑）。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings

logger = logging.getLogger(__name__)

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
        if self.device == "cuda":
            try:
                self.model = self.model.half()  # fp16 加速重排
            except Exception as e:
                logger.warning("bge 重排转 fp16 失败: %s", e)

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


class ApiReranker(Reranker):
    """通过私有推理节点(云 GPU)重排：POST {base}/rerank {query, documents} -> {results:[{index,relevance_score}]}。

    失败时降级为 RRF 原顺序(不回退本地 bge、不阻断回答,仅日志)。
    """

    def __init__(self, base: str | None = None, api_key: str | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.base = (base or s.rerank_api_base or "").rstrip("/")
        self.api_key = api_key or s.rerank_api_key or ""
        if not self.base:
            raise ValueError("RERANK_API_BASE 未设置(reranker_provider=api 时需指向推理节点)")

    def rerank(self, query: str, candidates: Sequence[dict]) -> list[dict]:
        if not candidates:
            return list(candidates)
        docs = [str(c.get("content", "")) for c in candidates]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Inference-Token"] = self.api_key
        try:
            with self._httpx.Client(timeout=60) as client:
                r = client.post(f"{self.base}/rerank", json={"query": query, "documents": docs, "top_n": len(docs)}, headers=headers)
                r.raise_for_status()
                results = r.json().get("results", [])
        except Exception as e:
            logger.warning("rerank 节点不可达,降级 RRF 顺序: %s", e)
            return list(candidates)  # 已按 RRF 融合排序
        score = {res["index"]: float(res["relevance_score"]) for res in results if "index" in res}
        out = []
        for i, c in enumerate(candidates):
            cc = dict(c)
            cc["rank_score"] = score.get(i, 0.0)
            out.append(cc)
        out.sort(key=lambda x: x["rank_score"], reverse=True)
        return out


def get_reranker(provider: str | None = None) -> Reranker:
    provider = provider or get_settings().reranker_provider
    if provider == "bge":
        return BgeReranker()
    if provider == "api":
        return ApiReranker()
    return FakeReranker()
