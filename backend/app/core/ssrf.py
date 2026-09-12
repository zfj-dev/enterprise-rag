"""`base_url` 的 SSRF 防护（票 33）：不许拿我方服务当内网跳板。

三条线，任一不合规就**拒绝**并给出人话原因（而不是等连接失败了才说）：
1. **协议**：默认只允许 `https`（本地开发可在配置里**显式**放行 `http`）；
2. **地址**：私网 / 回环 / 链路本地 / 保留 / 组播 / 未指定地址一律拒绝 —— IP 字面量直接判，
   域名则**解析后判解析出来的 IP**（否则 `evil.example.com` 指到 127.0.0.1 就绕过去了）；
3. **白名单**：配了就只放行表里的域名（精确或子域）。

解析器从外部注入（`Runtime.url_resolver`）：真实运行时用 socket，测试注入 stub —— 既诚实又离线可测。
"""
from __future__ import annotations

import ipaddress
import socket

from typing import Callable, Sequence
from urllib.parse import urlsplit

Resolver = Callable[[str], Sequence[str]]


def default_resolver(host: str) -> list[str]:
    """默认解析器：走系统 DNS。拿不到就返回空表（调用方按「验不了 → 拒绝」处理）。"""
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None)]
    except Exception:      # noqa: BLE001 —— 解析失败不该抛穿，交给调用方决定
        return []


def _norm_host(host: str) -> str:
    """主机名归一：小写、去首尾空白、去尾部点（`example.com.` 与 `example.com` 是同一个）。"""
    return str(host or "").strip().lower().rstrip(".")


def _ip_is_blocked(ip: str) -> bool:
    """这个 IP 是不是「不许连的那一类」。IPv4-mapped IPv6 也按 IPv4 判（`::ffff:127.0.0.1`）。

    认不出来的地址**保守当禁止** —— 名字说的是「不许连」，不是「确认是内网」。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def _host_allowed(host: str, allowed: Sequence[str]) -> bool:
    """白名单命中：精确匹配或它的子域（`api.example.com` 命中 `example.com`）。

    **不能**用 `endswith(a)` —— 那样 `evil-example.com` 也会命中 `example.com`。
    """
    host = _norm_host(host)
    return any(host == a or host.endswith("." + a) for a in allowed)


def check_base_url(raw: str, *, allowed_hosts: Sequence[str] = (), allow_insecure: bool = False,
                   resolve: Resolver | None = None) -> str | None:
    """校验用户自填的 `base_url`：合规返回 None，不合规返回**拒绝原因**。

    原因要写得让人一眼看出「是地址不合规」，而不是一个含糊的连接错误。

    **这是保存时的第一道**：真正发请求前会**再验一次**（Runtime.llm_for）—— 保存与使用之间
    域名可能被改指到内网（DNS rebinding）。残余窗口只到「这次查询」与「真连接」之间；
    要做绝就得把解析出来的 IP 钉住，那会破坏 TLS 证书校验，代价更大。
    """
    text = str(raw or "").strip()
    if not text:
        return "base_url 不能为空"
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        return "base_url 必须是 http(s) 地址"
    if parts.scheme != "https" and not allow_insecure:
        return "base_url 必须是 https —— 明文 http 只在本地开发时显式放行（BYOK_ALLOW_INSECURE）"
    host = _norm_host(parts.hostname or "")
    if not host:
        return "base_url 里没有主机名"

    allowed = [_norm_host(a) for a in (allowed_hosts or []) if str(a).strip()]
    if allowed and not _host_allowed(host, allowed):
        return "base_url 的主机 %s 不在白名单里（BYOK_ALLOWED_HOSTS）" % host

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return ("base_url 指向内网/回环地址（%s）—— 不许把本服务当内网跳板" % host
                if _ip_is_blocked(host) else None)

    # 域名：**解析后按 IP 判**，否则指到 127.0.0.1 的域名就绕过去了
    resolved = list((resolve or default_resolver)(host) or [])
    if not resolved:
        return "base_url 的主机 %s 解析不了 —— 验不了就不放行" % host
    for ip in resolved:
        if _ip_is_blocked(ip):
            return ("base_url 的主机 %s 解析到了内网/回环地址（%s）—— 不许把本服务当内网跳板"
                    % (host, ip))
    return None


def parse_allowed_hosts(raw: str) -> list[str]:
    """把配置里的白名单（逗号分隔）拆成小写域名表；空配置 = 不限主机。"""
    return [t.strip().lower().rstrip(".") for t in str(raw or "").split(",") if t.strip()]
