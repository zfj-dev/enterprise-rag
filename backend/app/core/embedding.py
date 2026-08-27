"""Embedding 抽象与实现。

- FakeEmbedding: 基于词袋哈希的确定性向量（无需模型/GPU），使 demo 与测试中向量检索可用。
- BgeEmbedding:  真实 bge-large-zh（惰性 import sentence_transformers，需 GPU，使用真机）。
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings


class EmbeddingModel(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FakeEmbedding(EmbeddingModel):
    """词袋哈希向量：相同/相近词更接近，L2 归一化。dim 可配（默认取 settings.embedding_dim）。"""

    def __init__(self, dim: int | None = None):
        self.dim = dim or get_settings().embedding_dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in str(text).lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class BgeEmbedding(EmbeddingModel):
    """真实 bge-large-zh（惰性加载 sentence-transformers）。"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # 惰性 import

        s = get_settings()
        self.model = SentenceTransformer(model_name or s.embedding_model, device=device or "cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=32).tolist()


def get_embedding(provider: str | None = None) -> EmbeddingModel:
    provider = provider or get_settings().embedding_provider
    if provider == "bge":
        return BgeEmbedding()
    return FakeEmbedding()
