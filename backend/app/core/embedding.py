"""Embedding 抽象与实现。

- FakeEmbedding: 字符级词袋向量（中文缩写/近义词可部分匹配；无需模型/GPU，用于演示/测试）。
- BgeEmbedding: 真实 bge-large-zh（本地 GPU，需 torch + sentence-transformers；宿主跑）。
- ApiEmbedding: 自建推理节点（协议自研，需要自己有 GPU 机器）。
- SiliconFlowEmbedding: 托管嵌入（OpenAI 兼容，**无 GPU 也能跑真实检索**）。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import SILICONFLOW_BASE, get_settings

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


class SiliconFlowEmbedding(EmbeddingModel):
    """托管嵌入（SiliconFlow；走 **OpenAI 兼容**的 `POST {base}/embeddings`）。

    与 `api` 并列：那边要你自己有一台 GPU 机器跑推理节点，这边**无 GPU 也能跑真实检索**
    —— 在线 Demo（CPU VPS）与本地跑评测都靠它。因为是 OpenAI 兼容的形状，
    换一家同样协议的服务只改 `EMBEDDING_API_BASE`。

    **失败一律抛**，不降级成空向量：返回一堆空向量等于「什么都没检索到」，
    看上去却像正常跑完 —— 那比报错更坏。
    """

    def __init__(self, base: str | None = None, api_key: str | None = None,
                 model: str | None = None, batch_size: int | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.base = (base or s.embedding_api_base or SILICONFLOW_BASE).rstrip("/")
        self.api_key = api_key or s.embedding_api_key or ""
        self.model = model or s.embedding_model
        self.batch_size = batch_size or s.embedding_batch_size
        self.dim = s.embedding_dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key   # 托管侧是 Bearer
        out: list[list[float]] = []
        with self._httpx.Client(timeout=120) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                r = client.post("%s/embeddings" % self.base,
                                json={"model": self.model, "input": batch}, headers=headers)
                r.raise_for_status()
                got = _ordered_embeddings(r.json(), len(batch))
                self._check_dim(got)
                out.extend(got)
        return out

    def _check_dim(self, vectors: list) -> None:
        """维度对不上要当场抛：`cosine` 用 `zip` 对比，长度不一时**在短的一侧截断**，
        算出来的相似度是错的却不报错 —— 检索悄悄变差，比直接失败更难查。"""
        bad = sorted({len(v) for v in vectors if len(v) != self.dim})
        if bad:
            raise RuntimeError(
                "嵌入维度 %s 与 EMBEDDING_DIM=%d 不一致 —— 相似度会算错。"
                "请把 EMBEDDING_DIM 设成 %s。" % (bad, self.dim, bad))


def _ordered_embeddings(body: dict, expected: int) -> list[list[float]]:
    """**按 `index` 还原顺序** —— OpenAI 兼容接口不保证 `data[]` 与 `input[]` 同序。

    照返回顺序取会把向量和文本错配；这种错不报错、只是检索悄悄变差，所以宁可在这里
    对不上就抛，也不按位置猜。
    """
    data = body.get("data") or []
    if len(data) != expected:
        raise RuntimeError("嵌入返回 %d 条、请求 %d 条，对不上 —— 拒绝按顺序猜"
                           % (len(data), expected))
    by_index: dict[int, list[float]] = {}
    for pos, item in enumerate(data):
        vec = item.get("embedding")
        # **空向量不能放行**：存进去之后 cosine 恒为 0，那个分块就永远检索不到了，
        # 而且全程不报错。宁可在这里炸。
        if not vec:
            raise RuntimeError("嵌入返回了空向量（index=%s）—— 不降级成空，"
                               "那会让这段内容静默地检索不到" % item.get("index", pos))
        by_index[int(item.get("index", pos))] = list(vec)
    if sorted(by_index) != list(range(expected)):
        raise RuntimeError("嵌入返回的 index 不是 0..%d，无法还原顺序" % (expected - 1))
    return [by_index[i] for i in range(expected)]


def get_embedding(provider: str | None = None) -> EmbeddingModel:
    provider = provider or get_settings().embedding_provider
    if provider == "bge":
        return BgeEmbedding()
    if provider == "api":
        return ApiEmbedding()
    if provider == "siliconflow":
        return SiliconFlowEmbedding()
    return FakeEmbedding()
