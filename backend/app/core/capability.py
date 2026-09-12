"""模型能力探测（票 34）：探上下文窗口与**是否支持 tool-calling**，供代理链路判断能不能用。

两条纪律：
- **探测失败一律按保守默认** —— 「不支持工具」，并标注原因是「没探到」。**绝不默认支持**：
  默认支持的话，用户接一个不支持工具的模型，代理会当场坏掉。
- **结果缓存**（按 base_url + model + 凭据摘要）—— 不为每次问答都探一次；换了 key 就重探。

上下文窗口这一项：OpenAI 兼容协议里**没有**这个标准字段，所以真实探测一律留 `None` ——
**不猜一个数**（真要就得各厂商单独适配，那是另一件事）。

此外它还有**第二个消费者：界面**（票 36 / #44）—— 用户填完自带 Key 要能知道
「连得上吗 / 支不支持工具」。为此加了两件事：`cached()` 只读已有结论（打开面板不打外网），
`probe_fresh()` 强制重探（「测试连通性」按钮点下去必须真探一次，返回旧缓存等于没测）。
界面这一侧**只做展示**：`probe()` 供代理降级判定的那套语义不受影响。

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

    def cached(self, base_url: str, api_key: str, model: str) -> ModelCapability | None:
        """**只看已有结论，不发网络请求** —— 界面打开时不该打一次外网。默认：没有缓存。"""
        return None

    def probe_fresh(self, base_url: str, api_key: str,
                    model: str) -> tuple[ModelCapability | None, str]:
        """**强制探一次**（面板上的「测试连通性」）—— 命中缓存也要重探。

        带缓存的实现在这里必须绕开读取：用户刚改完地址或密钥，点一下测试却看到上一次的
        旧结论，会让他以为改动没生效。默认实现本身不缓存，直接复用 `probe_report`。
        """
        return self.probe_report(base_url, api_key, model)

    def probe_report(self, base_url: str, api_key: str,
                     model: str) -> tuple[ModelCapability | None, str]:
        """给界面用：结论之外再带一句「为什么没结论」。

        `probe()` 的语义（票 34：探不到 → None → **不落缓存**、代理按保守默认降级）**一字不动**，
        这里只是把原因也带出来，好让界面说人话而不是笼统报个「失败」。
        """
        got = self.probe(base_url, api_key, model)
        return got, (u"" if got is not None else u"没探到（探测器未给出具体原因）")


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
        return self.probe_report(base_url, api_key, model)[0]

    def probe_report(self, base_url: str, api_key: str,
                     model: str) -> tuple[ModelCapability | None, str]:
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
            return None, "连不上（%s）—— 请检查地址与网络" % type(e).__name__
        if resp.status_code == 200:
            # 协议里没有「上下文窗口」这个标准字段 —— 探不到就留 None，**不猜一个数**
            return ModelCapability(supports_tools=True, context_window=None, source="probed",
                                   note="探到 provider 接受了带 tools 的请求"), ""
        if resp.status_code in (400, 422):
            # 请求被拒：**多半**是 tools 这个参数不被认，但也可能只是模型名写错了 ——
            # 所以口径写「无法据此确认，按保守默认处理」，而不是断言「确认不支持」
            return conservative_capability("探测被拒（HTTP %d）—— 无法据此确认工具支持，"
                                           "按保守默认处理" % resp.status_code), ""
        # 401/403/404/429 与 5xx：与「支不支持工具」**无关**（密钥不对 / 地址写错 / 被限流 /
        # 对方服务抖了），一律返回 None —— 不落缓存，下次再试，绝不把它记成结论。
        # 但**原因按状态码分开写**：界面要能告诉用户到底哪一步错了。
        return None, _reason_for(resp.status_code)


class CachedCapabilityProbe(CapabilityProbe):
    """给任意探测器加一层缓存（按 base_url + model）—— 不为每次问答都探一遍。

    **只缓存探到的结论**：探不到（`None`）不落缓存，下次还会再试 —— 一次网络抖动不该
    把某个用户的代理永久关掉。
    """

    def __init__(self, inner: CapabilityProbe):
        self._inner = inner
        self._cache: dict[tuple[str, str], ModelCapability] = {}

    def probe(self, base_url: str, api_key: str, model: str) -> ModelCapability | None:
        return self.probe_report(base_url, api_key, model)[0]

    def cached(self, base_url: str, api_key: str, model: str) -> ModelCapability | None:
        return self._cache.get(self._cache_key(base_url, api_key, model))

    def probe_report(self, base_url: str, api_key: str,
                     model: str) -> tuple[ModelCapability | None, str]:
        hit = self.cached(base_url, api_key, model)
        if hit is not None:
            return hit, ""
        return self.probe_fresh(base_url, api_key, model)

    def probe_fresh(self, base_url: str, api_key: str,
                    model: str) -> tuple[ModelCapability | None, str]:
        """**绕开缓存读**，探一次并刷新缓存 —— 「测试连通性」要的是当下这一下。"""
        got, reason = self._inner.probe_report(base_url, api_key, model)
        if got is not None:
            self._cache[self._cache_key(base_url, api_key, model)] = got
        return got, reason

    @staticmethod
    def _cache_key(base_url: str, api_key: str, model: str) -> tuple[str, str, str]:
        """缓存键带上凭据的**摘要**：换了 key（比如刚填错过一次）就该重新探，而不是沿用旧结论；
        也免得一个用户的结论被另一个用户（同一地址 + 同一模型）直接拿去用。"""
        digest = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:12]
        return (str(base_url).rstrip("/"), str(model), digest)


def _reason_for(status: int) -> str:
    """把探测失败的状态码翻成一句人话。**分开写**的原因：这些错法与「支不支持工具」无关，
    笼统报一句「探测失败」会让用户以为是自己模型不行。"""
    if status == 401:
        return "探测被拒：凭据不对（HTTP 401）—— 这与「支不支持工具」无关，请检查 Key"
    if status == 403:
        return "探测被拒：没有权限（HTTP 403）—— 请检查这个 Key 的授权范围"
    if status == 404:
        return "地址或模型名不对（HTTP 404）—— 请检查 base_url 与 model"
    if status == 429:
        return "被限流（HTTP 429）—— 稍后再试"
    if status >= 500:
        return "对方服务异常（HTTP %d）—— 稍后再试" % status
    return "意外的响应（HTTP %d）" % status


def capability_for(probe: CapabilityProbe, base_url: str, api_key: str,
                   model: str) -> ModelCapability:
    """探一次并**永远给一个结论**：探不到就按保守默认（并要求调用方把 note 说出来）。"""
    got = probe.probe(base_url, api_key, model)
    if got is not None:
        return got
    return conservative_capability("没探到该模型的能力 —— 按保守默认处理（视为不支持工具）")
