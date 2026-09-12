"""Reranker 抽象与实现（重排是检索质量最高 ROI 的一步）。

- FakeReranker: 词重叠分数（演示/测试）。
- BgeReranker:  真实 bge-reranker-large（本地 GPU，需 torch + transformers；宿主跑）。
- ApiReranker:  自建推理节点（协议自研，需要自己有 GPU 机器）。
- SiliconFlowReranker: 托管重排（Cohere 兼容，**无 GPU 也能跑**）。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import SILICONFLOW_BASE, get_settings

logger = logging.getLogger(__name__)

_RELEVANCE_OFFSET = 0.15


def _degrade_or_raise(info: dict | None, err: Exception, what: str) -> None:
    """重排不可达时的**唯一**处置：严格模式直接抛，否则记下这次降级了。

    严格模式是给评测用的 —— 那边要的是「这一段没跑成」，而不是把 RRF 原顺序的数字
    说成「重排已跑」。线上默认不严格：重排挂掉不该把问答也打断。
    降级的事实写进 `info` 出参 —— 只写日志等于只有看日志的人知道。
    """
    if get_settings().rerank_strict:
        raise err
    logger.warning("%s不可达，降级 RRF 顺序：%s", what, err)
    if info is not None:
        info["degraded"] = True
        info["note"] = "重排不可达，已降级为 RRF 原顺序：%s" % err


def _apply_scores(candidates: Sequence[dict], results: Sequence[dict],
                  info: dict | None = None) -> list[dict]:
    """把 `{index, relevance_score}` 的结果落回候选并排序 —— 自建节点与托管服务共用一份。

    没被评分的候选记 **0 分**（不猜它其实相关）；但它们**与「评了 0 分」不是一回事** ——
    前者是「不知道」，后者是「确实不相关」。所以另外数一个 `unscored` 交给调用方，
    别让两者在报告里混成一个数。
    """
    score = {res["index"]: float(res["relevance_score"]) for res in results if "index" in res}
    out = []
    for i, c in enumerate(candidates):
        cc = dict(c)
        cc["rank_score"] = score.get(i, 0.0)
        out.append(cc)
    out.sort(key=lambda x: x["rank_score"], reverse=True)
    if info is not None:
        missing = sum(1 for i in range(len(candidates)) if i not in score)
        if missing:
            info["unscored"] = missing
    return out


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[dict], *,
               info: dict | None = None) -> list[dict]:
        """重排候选。`info` 是**出参**：把「降级了 / 有候选没被评分」这类
        **不该被当成结论**的事实写进去，由调用方带走。

        与 `HybridRetriever.retrieve` 的 `timings` 同一个手法 —— 不改返回值形状，
        也不需要在线程间共享可变状态（运行时是单例、请求可能并发）。
        """


class FakeReranker(Reranker):
    def rerank(self, query: str, candidates: Sequence[dict], *,
               info: dict | None = None) -> list[dict]:
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

    def rerank(self, query: str, candidates: Sequence[dict], *,
               info: dict | None = None) -> list[dict]:
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

    def rerank(self, query: str, candidates: Sequence[dict], *,
               info: dict | None = None) -> list[dict]:
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
            _degrade_or_raise(info, e, "rerank 节点")
            return list(candidates)  # 已按 RRF 融合排序
        return _apply_scores(candidates, results, info)


class SiliconFlowReranker(Reranker):
    """托管重排（SiliconFlow；Cohere 兼容的扁平 `POST {base}/rerank`）。

    比 `api` 多了两样：请求体带 `model`，鉴权走 `Authorization: Bearer`。
    **失败降级为 RRF 原顺序**（与 `api` 一致）—— 重排挂掉不该把回答也打断。
    """

    def __init__(self, base: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.base = (base or s.rerank_api_base or SILICONFLOW_BASE).rstrip("/")
        self.api_key = api_key or s.rerank_api_key or ""
        self.model = model or s.reranker_model

    def rerank(self, query: str, candidates: Sequence[dict], *,
               info: dict | None = None) -> list[dict]:
        if not candidates:
            return list(candidates)
        docs = [str(c.get("content", "")) for c in candidates]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        try:
            with self._httpx.Client(timeout=60) as client:
                r = client.post("%s/rerank" % self.base,
                                json={"model": self.model, "query": query,
                                      "documents": docs, "top_n": len(docs)}, headers=headers)
                r.raise_for_status()
                results = r.json().get("results", [])
        except Exception as e:
            _degrade_or_raise(info, e, "托管重排")
            return list(candidates)          # 已按 RRF 融合排序
        return _apply_scores(candidates, results, info)


def get_reranker(provider: str | None = None) -> Reranker:
    provider = provider or get_settings().reranker_provider
    if provider == "bge":
        return BgeReranker()
    if provider == "api":
        return ApiReranker()
    if provider == "siliconflow":
        return SiliconFlowReranker()
    return FakeReranker()
