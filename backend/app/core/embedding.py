"""Embedding 抽象与实现。

- FakeEmbedding: 字符级词袋向量（中文缩写/近义词可部分匹配；无需模型/GPU，用于演示/测试）。
- BgeEmbedding: 真实 bge-large-zh（本地 GPU，需 torch + sentence-transformers；宿主跑）。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings

logger = logging.getLogger(__name__)

_ASCII = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]")


class EmbeddingModel(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FakeEmbedding(EmbeddingModel):
    def __init__(self, dim: int | None = None):
        self.dim = dim or get_settings().embedding_dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _CJK.findall(str(text).lower()) + _ASCII.findall(str(text).lower())

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class BgeEmbedding(EmbeddingModel):
    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        s = get_settings()
        dev = device or s.embedding_device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        print(f"[bge] embedding device={dev} model={model_name or s.embedding_model}")
        self.model = SentenceTransformer(model_name or s.embedding_model, device=dev)
        if dev == "cuda":
            try:
                self.model = self.model.half()  # fp16 加速嵌入
            except Exception as e:
                logger.warning("bge 嵌入转 fp16 失败: %s", e)
        try:
            self.dim = self.model.get_embedding_dimension()   # 新版 API
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()  # 旧版兼容

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=32).tolist()


class ApiEmbedding(EmbeddingModel):
    """通过私有推理节点(云 GPU)嵌入：POST {base}/embed {texts:[...]} -> {vectors:[[...]]}。

    批量(embedding_batch_size)+ 超时;节点不可达抛异常(已选纯云端,不回退本地 bge)。
    """

    def __init__(self, base: str | None = None, api_key: str | None = None, batch_size: int | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.base = (base or s.embedding_api_base or "").rstrip("/")
        self.api_key = api_key or s.embedding_api_key or ""
        self.batch_size = batch_size or s.embedding_batch_size
        self.dim = s.embedding_dim
        if not self.base:
            raise ValueError("EMBEDDING_API_BASE 未设置(embedding_provider=api 时需指向推理节点)")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Inference-Token"] = self.api_key
        out: list[list[float]] = []
        with self._httpx.Client(timeout=120) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                r = client.post(f"{self.base}/embed", json={"texts": batch}, headers=headers)
                r.raise_for_status()
                out.extend(r.json().get("vectors", []))
        return out


def get_embedding(provider: str | None = None) -> EmbeddingModel:
    provider = provider or get_settings().embedding_provider
    if provider == "bge":
        return BgeEmbedding()
    if provider == "api":
        return ApiEmbedding()
    return FakeEmbedding()
