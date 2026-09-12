"""余额提醒（票 35）：**能查则查，查不到就明说** —— 绝不拿我方用量冒充厂商余额。

三种结果分得清清楚楚，任何一种都不会被说成「余额 0」：
- 查到了 → 展示**厂商的真实余额**；低于阈值**只提醒、不拦截**；
- 厂商**没有**这个接口（404/405）→ 明写「该厂商不支持余额查询」；
- 查询**失败**（网络 / 凭据不对 / 5xx）→ 明写「查不到」，并说明是哪一类。

**我方累计用量是另一个维度**：它照常展示（可审计），但那是「我方统计用量」，不是余额 ——
两个数字各自的来源必须分别标明。
"""
from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VendorBalance:
    """厂商余额的查询结论。`available=False` 时 `amount` 一定是 None（不拿 0 顶替）。"""

    available: bool
    amount: float | None = None
    currency: str = ""
    note: str = ""            # 这个数字**从哪来** / 为什么没有
    alert: str = ""           # 低于阈值时的提醒文案（**只提醒**）


class BalanceProbe(ABC):
    """查厂商余额。查不到一律返回 `available=False` 并写明原因，**绝不抛穿、绝不编数**。"""

    @abstractmethod
    def fetch(self, base_url: str, api_key: str) -> VendorBalance: ...


class OpenAICompatBalanceProbe(BalanceProbe):
    """按 OpenAI 兼容的老办法试一次 `/user/balance`。

    DeepSeek 有这个端点（返回 `balance_infos`），多数厂商没有 —— 这一点**如实区分**：
    没有这个接口 ≠ 查询失败 ≠ 余额为 0。
    """

    def __init__(self, timeout: float | None = None):
        from app.config import get_settings

        self._timeout = timeout or get_settings().byok_request_timeout_seconds

    def fetch(self, base_url: str, api_key: str) -> VendorBalance:
        import httpx

        url = balance_url(base_url)
        headers = {"Authorization": "Bearer %s" % api_key}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers=headers)
        except Exception as e:      # noqa: BLE001 —— 查不到不是错误，是「没这个数字」
            logger.warning("余额查询失败（如实报告）：%s", type(e).__name__)
            return VendorBalance(available=False, note="余额查询失败（网络不通），稍后再试")
        if resp.status_code in (404, 405):
            return VendorBalance(available=False, note="该厂商不支持余额查询")
        if resp.status_code in (401, 403):
            return VendorBalance(available=False, note="余额查询被拒（凭据不对），不是「不支持」")
        if resp.status_code != 200:
            return VendorBalance(available=False,
                                 note="余额查询失败（HTTP %d），不是「不支持」" % resp.status_code)
        try:
            body = resp.json()
        except Exception as e:      # noqa: BLE001 —— 对方回了个不是 JSON 的 200，也是「查不到」
            logger.warning("余额返回体不是 JSON（如实报告）：%s", type(e).__name__)
            return VendorBalance(available=False, note="余额返回体不是 JSON —— 这不是「不支持」，是查不动")
        amount, currency = _parse_balance(body)
        if amount is None:
            return VendorBalance(available=False, note="该厂商没有返回可识别的余额字段")
        return VendorBalance(available=True, amount=amount, currency=currency,
                             note="厂商余额（%s /user/balance）" % _vendor_of(base_url))


def balance_url(base_url: str) -> str:
    """余额端点挂在**站根**：DeepSeek 是 `https://api.deepseek.com/user/balance`，
    而 base_url 通常带 `/v1`（如 `https://api.deepseek.com/v1`）—— 直接拼会变成
    `/v1/user/balance`，撞 404，被误报成「该厂商不支持余额查询」。
    """
    parts = urlsplit(str(base_url or ""))
    return "%s://%s/user/balance" % (parts.scheme or "https", parts.netloc or parts.path.strip("/"))


def _vendor_of(base_url: str) -> str:
    return urlsplit(str(base_url or "")).hostname or "厂商"


def _parse_balance(body) -> tuple[float | None, str]:
    """从 `balance_infos` 里取余额（DeepSeek 的形状）；取不到就 (None, "")。"""
    infos = (body or {}).get("balance_infos") if isinstance(body, dict) else None
    for info in infos or []:
        if not isinstance(info, dict):
            continue
        for key in ("total_balance", "balance", "available_balance"):
            try:
                return float(info[key]), str(info.get("currency") or "")
            except (KeyError, TypeError, ValueError):
                continue
    return None, ""


def with_alert(balance: VendorBalance, threshold: float | None) -> VendorBalance:
    """低于阈值时补一句提醒 —— **只提醒，不拦截**（用户花的是自己的钱）。"""
    if not balance.available or not threshold or balance.amount is None:
        return balance
    if balance.amount >= threshold:
        return balance
    # 用 replace 而不是手抄一遍字段：将来加字段不会静默丢掉
    return replace(balance, alert="厂商余额 %.2f %s 已低于提醒线 %.2f —— 只提醒，不拦截"
                                  % (balance.amount, balance.currency, threshold))
