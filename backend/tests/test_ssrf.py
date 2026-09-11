"""base_url 的 SSRF 防护（票 33 / #40）：不许拿本服务当内网跳板。

三条线：非 https 拒（可显式放行本地开发）、私网/回环/链路本地/保留网段拒、
白名单配了就只放行表里的域名。域名**解析后按 IP 判** —— 否则指到 127.0.0.1 的域名就绕过去了。
"""
from __future__ import annotations

import pytest

from app.core.ssrf import check_base_url, parse_allowed_hosts
from tests.helpers import register_and_kb

PUBLIC = "93.184.216.34"          # 一个公网 IP（example.com 的老地址）


def _resolves_to(*ips):
    return lambda host: list(ips)


# ---------- 私网 / 回环 / 保留 ----------

@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.9.9",      # 回环 + 私网
    "169.254.169.254",                                          # 链路本地（云元数据那个）
    "0.0.0.0", "224.0.0.1", "240.0.0.1",                        # 未指定 / 组播 / 保留
    "[::1]", "[fe80::1]", "[fc00::1]",                          # IPv6 回环 / 链路本地 / 私有（URL 里带方括号）
    "[::ffff:127.0.0.1]",                                       # IPv4-mapped 也得挡
])
def test_internal_addresses_are_rejected(host):
    reason = check_base_url("https://%s/v1" % host, resolve=_resolves_to(PUBLIC))

    assert reason and "内网/回环" in reason          # 说清是地址不合规，不是「连不上」


def test_a_public_address_is_allowed():
    assert check_base_url("https://%s/v1" % PUBLIC, resolve=_resolves_to(PUBLIC)) is None


def test_a_hostname_pointing_at_an_internal_ip_is_rejected():
    """关键一条：域名不解析就直接放行的话，`evil.example.com` 指到 127.0.0.1 就绕过去了。"""
    reason = check_base_url("https://evil.example.com/v1", resolve=_resolves_to("127.0.0.1"))

    assert reason and "内网/回环" in reason


def test_a_hostname_that_cannot_be_resolved_is_refused():
    """验不了就不放行 —— 别把「解析不出来」当成「大概是好的」。"""
    reason = check_base_url("https://nope.invalid/v1", resolve=_resolves_to())

    assert reason and "解析不了" in reason


def test_a_hostname_resolving_to_a_public_ip_is_allowed():
    assert check_base_url("https://api.example.com/v1", resolve=_resolves_to(PUBLIC)) is None


# ---------- 协议 ----------

def test_plain_http_is_refused_unless_explicitly_allowed():
    assert "https" in check_base_url("http://api.example.com/v1", resolve=_resolves_to(PUBLIC))
    assert check_base_url("http://api.example.com/v1", allow_insecure=True,
                          resolve=_resolves_to(PUBLIC)) is None


def test_an_unexpected_scheme_is_refused():
    assert check_base_url("ftp://api.example.com/v1", resolve=_resolves_to(PUBLIC))


def test_an_empty_or_hostless_url_is_refused():
    assert check_base_url("")
    assert check_base_url("https:///v1", resolve=_resolves_to(PUBLIC))


# ---------- 白名单 ----------

def test_the_allowlist_accepts_exact_hosts_and_their_subdomains():
    allowed = parse_allowed_hosts("example.com, api.other.com")

    assert check_base_url("https://example.com/v1", allowed_hosts=allowed,
                          resolve=_resolves_to(PUBLIC)) is None
    assert check_base_url("https://a.example.com/v1", allowed_hosts=allowed,
                          resolve=_resolves_to(PUBLIC)) is None
    assert check_base_url("https://api.other.com/v1", allowed_hosts=allowed,
                          resolve=_resolves_to(PUBLIC)) is None


def test_the_allowlist_is_not_fooled_by_a_suffix_lookalike():
    """`evil-example.com` 不是 `example.com` 的子域 —— 用 endswith 裸判就会放它进来。"""
    allowed = parse_allowed_hosts("example.com")

    assert check_base_url("https://evil-example.com/v1", allowed_hosts=allowed,
                          resolve=_resolves_to(PUBLIC))
    assert check_base_url("https://example.com.evil.com/v1", allowed_hosts=allowed,
                          resolve=_resolves_to(PUBLIC))


def test_the_allowlist_refuses_everything_else():
    reason = check_base_url("https://evil.com/v1", allowed_hosts=parse_allowed_hosts("example.com"),
                            resolve=_resolves_to(PUBLIC))

    assert reason and "白名单" in reason


def test_an_empty_allowlist_means_no_host_restriction():
    assert parse_allowed_hosts("") == []
    assert check_base_url("https://anything.example/v1", allowed_hosts=[],
                          resolve=_resolves_to(PUBLIC)) is None


# ---------- 接进接口 ----------

def _seeded(client, name):
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, name)
    return H, uid, get_runtime()


def test_the_endpoint_refuses_an_internal_address_with_a_clear_reason(client):
    """https + 内网地址：撞的是**地址**那条线，不是协议那条。"""
    H, uid, rt = _seeded(client, "ssrf_internal")
    rt.url_resolver = _resolves_to(PUBLIC)

    r = client.put("/api/v1/llm/config", headers=H,
                   json={"base_url": "https://169.254.169.254/v1", "key": "sk-x-0001", "model": "m"})

    assert r.status_code == 400 and "内网/回环" in r.json()["detail"]
    assert client.get("/api/v1/llm/config", headers=H).json()["configured"] is False


def test_the_endpoint_refuses_plain_http_over_an_internal_address(client):
    """明文 http + 内网：先撞协议那条，原因照样说得清（不是含糊的连接失败）。"""
    H, uid, rt = _seeded(client, "ssrf_http")
    rt.url_resolver = _resolves_to(PUBLIC)

    r = client.put("/api/v1/llm/config", headers=H,
                   json={"base_url": "http://169.254.169.254/v1", "key": "sk-x-0004", "model": "m"})

    assert r.status_code == 400 and "https" in r.json()["detail"]


# ---------- 保存之后域名改指内网（DNS rebinding）：用之前要再验一次 ----------

class _NamedStubFactory:
    def __init__(self):
        self.built = []

    def build(self, cfg):
        self.built.append(cfg)
        return "LLM(own)"


def test_a_base_url_that_starts_pointing_inward_later_falls_back_to_the_global(client):
    """保存时解析到公网、之后被改指到内网 —— **用之前必须再验一次**，否则就成了内网跳板。"""
    H, uid, rt = _seeded(client, "ssrf_rebind")
    rt.url_resolver = _resolves_to(PUBLIC)
    rt.llm_factory = _NamedStubFactory()

    ok = client.put("/api/v1/llm/config", headers=H,
                    json={"base_url": "https://api.example.com/v1", "key": "sk-x-0005", "model": "own"})
    assert ok.status_code == 200
    assert rt.llm_for(uid) == "LLM(own)"               # 这时还是安全的

    rt.url_resolver = _resolves_to("10.0.0.7")         # 域名被改指到内网
    assert rt.llm_for(uid) is rt.llm                    # 复查不通过 → 回落服务端全局


# ---------- 请求超时上限：封顶要真的传下去 ----------

def test_the_configured_timeout_reaches_the_cloud_llm():
    """「请求超时上限」得真的传到模型实例上，不然配了也白配。"""
    from app.core.byok import LLMConfig, OpenAICompatLLMFactory

    cfg = LLMConfig(base_url="https://api.example.com/v1", api_key="k", model="m")

    assert OpenAICompatLLMFactory(timeout=7.5).build(cfg).timeout == 7.5
    assert OpenAICompatLLMFactory().build(cfg).timeout == 60.0            # 没配就用默认
    assert OpenAICompatLLMFactory(timeout=0).build(cfg).timeout == 0.0    # 0 不静默变 60


def test_the_endpoint_refuses_a_private_host_behind_a_domain(client):
    H, uid, rt = _seeded(client, "ssrf_domain")
    rt.url_resolver = _resolves_to("10.0.0.7")

    r = client.put("/api/v1/llm/config", headers=H,
                   json={"base_url": "https://sneaky.example.com/v1", "key": "sk-x-0002", "model": "m"})

    assert r.status_code == 400 and "内网/回环" in r.json()["detail"]
    assert client.get("/api/v1/llm/config", headers=H).json()["configured"] is False


def test_the_endpoint_accepts_a_compliant_address(client):
    H, uid, rt = _seeded(client, "ssrf_ok")
    rt.url_resolver = _resolves_to(PUBLIC)

    r = client.put("/api/v1/llm/config", headers=H,
                   json={"base_url": "https://api.example.com/v1", "key": "sk-x-0003", "model": "m"})

    assert r.status_code == 200 and r.json()["configured"] is True
