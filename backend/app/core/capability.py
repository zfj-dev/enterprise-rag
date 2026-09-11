"""模型能力探测（票 34）：探上下文窗口与**是否支持 tool-calling**，供代理链路判断能不能用。

两条纪律：
- **探测失败一律按保守默认** —— 「不支持工具」，并标注原因是「没探到」。**绝不默认支持**：
  默认支持的话，用户接一个不支持工具的模型，代理会当场坏掉。
- **结果缓存**（按 base_url + model + 凭据摘要）—— 不为每次问答都探一次；换了 key 就重探。

上下文窗口这一项：OpenAI 兼容协议里**没有**这个标准字段，所以真实探测一律留 `None` ——
**不猜一个数**（真要就得各厂商单独适配，那是另一件事）。

探测要发网络请求，所以从外部注入：真实实现走 OpenAI 兼容接口，测试注入 stub。
"""
from __future__ import annotations

import hashlib
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapability:
    """探测结论。`source` 说清这是**探到的**还是**保守默认**。"""

    supports_tools: bool
    context_window: int | None = None
    source: str = "conservative"      # probed / conservative
    note: str = ""


def conservative_capability(note: str) -> ModelCapability:
    """探测不到时的落点：**不支持工具** + 写明为什么。"""
    return ModelCapability(supports_tools=False, context_window=None,
                           source="conservative", note=note)


class CapabilityProbe(ABC):
    """探一个模型的能力。探测失败一律返回 None —— 由调用方按保守默认处理。"""

    @abstractmethod
    def probe(self, base_url: str, api_key: str, model: str) -> ModelCapability | None: ...


_PROBE_TOOL = [{
    "type": "function",
    "function": {"name": "ping", "description": "连通性探测用，不必实现",
                 "parameters": {"type": "object", "properties": {}}},
}]


class OpenAICompatCapabilityProbe(CapabilityProbe):
    """按 OpenAI 兼容协议探：带一个极小的 tools 发一次非流式请求。

    代理**能**用它的判据是「provider 收了 tools 且没有报错」——不支持工具的 provider 通常
    直接 4xx。上下文窗口多数 provider 不暴露，探不到就留 None（不猜一个数）。
    """

    def __init__(self, timeout: float | None = None):
        s = get_settings()
        self._timeout = timeout or s.byok_request_timeout_seconds

    def probe(self, base_url: str, api_key: str, model: str) -> ModelCapability | None:
        import httpx

        url = "%s/chat/completions" % str(base_url).rstrip("/")
        payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
                   "max_tokens": 1, "tools": _PROBE_TOOL}
        headers = {"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
        except Exception as e:      # noqa: BLE001 —— 探测失败不是错误，是「没探到」
            logger.warning("能力探测失败（按保守默认处理）：%s", type(e).__name__)
            return None
        if resp.status_code == 200:
            # 协议里没有「上下文窗口」这个标准字段 —— 探不到就留 None，**不猜一个数**
            return ModelCapability(supports_tools=True, context_window=None, source="probed",
                                   note="探到 provider 接受了带 tools 的请求")
        if resp.status_code in (400, 422):
            # 请求被拒：**多半**是 tools 这个参数不被认，但也可能只是模型名写错了 ——
            # 所以口径写「无法据此确认，按保守默认处理」，而不是断言「确认不支持」
            return conservative_capability("探测被拒（HTTP %d）—— 无法据此确认工具支持，"
                                           "按保守默认处理" % resp.status_code)
        # 401/403/404/429 与 5xx：与「支不支持工具」**无关**（密钥不对 / 地址写错 / 被限流 /
        # 对方服务抖了），一律返回 None —— 不落缓存，下次再试，绝不把它记成结论
        return None


class CachedCapabilityProbe(CapabilityProbe):
    """给任意探测器加一层缓存（按 base_url + model）—— 不为每次问答都探一遍。

    **只缓存探到的结论**：探不到（`None`）不落缓存，下次还会再试 —— 一次网络抖动不该
    把某个用户的代理永久关掉。
    """

    def __init__(self, inner: CapabilityProbe):
        self._inner = inner
        self._cache: dict[tuple[str, str], ModelCapability] = {}

    def probe(self, base_url: str, api_key: str, model: str) -> ModelCapability | None:
        key = self._cache_key(base_url, api_key, model)
        if key in self._cache:
            return self._cache[key]
        got = self._inner.probe(base_url, api_key, model)
        if got is not None:
            self._cache[key] = got
        return got

    @staticmethod
    def _cache_key(base_url: str, api_key: str, model: str) -> tuple[str, str, str]:
        """缓存键带上凭据的**摘要**：换了 key（比如刚填错过一次）就该重新探，而不是沿用旧结论；
        也免得一个用户的结论被另一个用户（同一地址 + 同一模型）直接拿去用。"""
        digest = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:12]
        return (str(base_url).rstrip("/"), str(model), digest)


def capability_for(probe: CapabilityProbe, base_url: str, api_key: str,
                   model: str) -> ModelCapability:
    """探一次并**永远给一个结论**：探不到就按保守默认（并要求调用方把 note 说出来）。"""
    got = probe.probe(base_url, api_key, model)
    if got is not None:
        return got
    return conservative_capability("没探到该模型的能力 —— 按保守默认处理（视为不支持工具）")
