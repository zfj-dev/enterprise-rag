"""模型能力探测 + 代理降级（票 34 / #41）：探不到就按保守默认，绝不默认「支持工具」。

探测要发网络请求，所以从外部注入：stub 探测器断言调用次数，真实实现只在真机上跑。
"""
from __future__ import annotations

import httpx
import pytest

from app.core.capability import (CapabilityProbe, CachedCapabilityProbe, ModelCapability,
                                 OpenAICompatCapabilityProbe, capability_for,
                                 conservative_capability)
from tests.helpers import offline_resolver as _resolver, register_and_kb, sse_events

UNSUPPORTED = ModelCapability(supports_tools=False, source="probed",
                              note="探测时 provider 拒绝了带 tools 的请求（HTTP 400）")


class StubProbe(CapabilityProbe):
    """按脚本返回结论；记录被调了几次（缓存用例要断言它）。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def probe(self, base_url, api_key, model):
        self.calls += 1
        return self._results.pop(0) if self._results else None


# ---------- 保守默认 ----------

def test_the_conservative_default_never_claims_tool_support():
    """探不到时的落点：**不支持工具**，并写明为什么 —— 默认「支持」会让代理当场坏掉。"""
    cap = conservative_capability("网络挂了")

    assert cap.supports_tools is False and cap.source == "conservative"
    assert "网络挂了" in cap.note


def test_probing_failure_yields_the_conservative_default():
    probe = StubProbe([None])                      # 探不到

    cap = capability_for(probe, "https://a/v1", "k", "m")

    assert cap.supports_tools is False and "没探到" in cap.note


def test_probing_success_is_passed_through_with_its_caliber():
    probe = StubProbe([ModelCapability(supports_tools=True, context_window=128000, source="probed",
                                       note="探到 provider 接受了带 tools 的请求")])

    cap = capability_for(probe, "https://a/v1", "k", "m")

    assert cap.supports_tools is True and cap.context_window == 128000 and cap.source == "probed"


# ---------- 缓存 ----------

def test_the_probe_is_cached_per_endpoint_and_model():
    """缓存按 (base_url, model)：第二次问同一个模型不该再探一遍。"""
    inner = StubProbe([ModelCapability(supports_tools=True, source="probed", note="ok")])
    probe = CachedCapabilityProbe(inner)

    a = capability_for(probe, "https://a/v1", "k", "m1")
    b = capability_for(probe, "https://a/v1", "k", "m1")     # 命中缓存
    c = capability_for(probe, "https://a/v1", "k", "m2")     # 换了模型 → 再探一次

    assert a.supports_tools is True and b.supports_tools is True
    assert c.supports_tools is False                        # 脚本用尽 → 探不到 → 保守默认
    assert inner.calls == 2                                 # 缓存省掉了第二次


def test_a_failed_probe_is_not_cached():
    """一次网络抖动不该把某个用户的代理**永久**关掉 —— 失败不落缓存，下次还会再试。"""
    inner = StubProbe([None, ModelCapability(supports_tools=True, source="probed", note="ok")])
    probe = CachedCapabilityProbe(inner)

    assert capability_for(probe, "https://a/v1", "k", "m").supports_tools is False
    assert capability_for(probe, "https://a/v1", "k", "m").supports_tools is True
    assert inner.calls == 2


def test_a_trailing_slash_does_not_split_the_cache():
    inner = StubProbe([ModelCapability(supports_tools=True, source="probed", note="ok")])
    probe = CachedCapabilityProbe(inner)

    capability_for(probe, "https://a/v1", "k", "m")
    capability_for(probe, "https://a/v1/", "k", "m")

    assert inner.calls == 1


# ---------- 真实探测器：按 HTTP 结果判 ----------

def _stub_httpx(monkeypatch, *, status: int | None, raises: bool = False):
    class Resp:
        def __init__(self): self.status_code = status

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, *a, **k):
            if raises:
                raise httpx.ConnectError("连不上")
            return Resp()

    monkeypatch.setattr(httpx, "Client", Client)


def test_a_provider_that_accepts_tools_is_marked_as_supporting_them(monkeypatch):
    _stub_httpx(monkeypatch, status=200)

    cap = OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m")

    assert cap.supports_tools is True and cap.source == "probed"


def test_a_provider_that_rejects_the_tools_request_is_marked_unsupported(monkeypatch):
    """400/422：请求被拒 —— 按保守默认，但口径写「无法据此确认」而不是「确认不支持」。"""
    _stub_httpx(monkeypatch, status=400)

    cap = OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m")

    assert cap.supports_tools is False
    assert "400" in cap.note and "无法据此确认" in cap.note


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_statuses_unrelated_to_tool_support_are_not_recorded(monkeypatch, status):
    """密钥不对 / 地址写错 / 被限流 —— 这些都**与「支不支持工具」无关**，不许记成结论。

    记了的话，一次填错 key 就会把代理**永久**关掉，而且报出来的原因是错的。
    """
    _stub_httpx(monkeypatch, status=status)

    assert OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m") is None


def test_such_a_status_is_not_cached_so_the_next_attempt_retries():
    inner = StubProbe([None, ModelCapability(supports_tools=True, source="probed", note="ok")])
    probe = CachedCapabilityProbe(inner)

    assert capability_for(probe, "https://a/v1", "k", "m").supports_tools is False
    assert capability_for(probe, "https://a/v1", "k", "m").supports_tools is True
    assert inner.calls == 2


def test_rotating_the_key_re_probes_instead_of_reusing_the_old_conclusion():
    """刚填错过一次 key、改对了 —— 不该沿用「不支持工具」那个旧结论。"""
    inner = StubProbe([UNSUPPORTED, ModelCapability(supports_tools=True, source="probed", note="ok")])
    probe = CachedCapabilityProbe(inner)

    assert capability_for(probe, "https://a/v1", "bad-key", "m").supports_tools is False
    assert capability_for(probe, "https://a/v1", "good-key", "m").supports_tools is True
    assert inner.calls == 2


def test_the_real_probe_does_not_guess_a_context_window(monkeypatch):
    """协议里没有这个标准字段 —— 拿不到就留 None，**不猜一个数**。"""
    _stub_httpx(monkeypatch, status=200)

    cap = OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m")

    assert cap.context_window is None and cap.supports_tools is True


def test_a_server_error_is_not_recorded_as_unsupported(monkeypatch):
    """5xx 是**对方服务的问题**，不是「这个模型不支持工具」—— 别把它当结论记下来。"""
    _stub_httpx(monkeypatch, status=503)

    assert OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m") is None


def test_a_network_failure_is_not_recorded_as_unsupported(monkeypatch):
    _stub_httpx(monkeypatch, status=None, raises=True)

    assert OpenAICompatCapabilityProbe().probe("https://a/v1", "k", "m") is None


# ---------- 接进问答链路：不支持工具时降级 ----------

def _seeded(client, name: str, *, probe):
    from app.api.deps import get_runtime
    from app.config import get_settings
    from app.core.byok import InMemoryUserLLMConfigStore, LLMConfig
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = get_runtime()
    rt.url_resolver = _resolver()
    rt.llm_factory = _NoopFactory()
    store = InMemoryUserLLMConfigStore()
    store.set(uid, LLMConfig(base_url="https://api.example.com/v1", api_key="sk-own", model="my-model"))
    rt.user_llm_config_store = store
    rt.capability_probe = probe
    return H, uid, kb, user, db, rt, get_settings()


class _NoopFactory:
    def build(self, cfg):
        from app.core.llm import FakeLLM

        return FakeLLM()


def _stub_agent(monkeypatch, called):
    import app.services.chat_service as chat_service

    monkeypatch.setattr(chat_service, "_run_agent_for",
                        lambda *a, **k: called.append(1) or {
                            "answer": "代理给的答案", "sources": [], "steps": [], "latency": {},
                            "stopped": "answered",
                            "trace": {"self_check": "skipped", "citation_coverage": None}})


def test_an_unsupported_model_degrades_to_a_single_step_answer(client, monkeypatch):
    """不支持工具 → 代理**不跑**、降级为单步回答、把原因写出来，且**不抛错**。"""
    import app.services.chat_service as chat_service

    H, uid, kb, user, db, rt, conf = _seeded(client, "cap_none", probe=StubProbe([UNSUPPORTED]))
    called: list = []
    try:
        monkeypatch.setattr(conf, "agent_enabled", True)
        _stub_agent(monkeypatch, called)

        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
    finally:
        db.close()

    assert called == []                                   # 代理一次都没跑
    assert out["answer"]                                  # 但回答照常给（降级而不是报错）
    assert "不支持工具调用" in out["trace"]["agent_skipped"]
    assert "my-model" in out["trace"]["agent_skipped"]


def test_a_supported_model_still_runs_the_agent(client, monkeypatch):
    """回归：探到支持工具时，代理走正常多步路径。"""
    import app.services.chat_service as chat_service

    ok = ModelCapability(supports_tools=True, source="probed", note="ok")
    H, uid, kb, user, db, rt, conf = _seeded(client, "cap_yes", probe=StubProbe([ok]))
    called: list = []
    try:
        monkeypatch.setattr(conf, "agent_enabled", True)
        _stub_agent(monkeypatch, called)

        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
    finally:
        db.close()

    assert called == [1]
    assert out["answer"] == "代理给的答案"
    assert out["trace"].get("agent_skipped") is None


def test_a_user_without_their_own_model_is_not_probed(client, monkeypatch):
    """没配自带模型 = 走服务端全局（今天的行为）—— 不做探测，也不该被挡住。"""
    import app.services.chat_service as chat_service
    from app.core.byok import InMemoryUserLLMConfigStore

    H, uid, kb, user, db, rt, conf = _seeded(client, "cap_global", probe=StubProbe([UNSUPPORTED]))
    rt.user_llm_config_store = InMemoryUserLLMConfigStore()      # 谁都没配
    called: list = []
    try:
        monkeypatch.setattr(conf, "agent_enabled", True)
        _stub_agent(monkeypatch, called)

        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
    finally:
        db.close()

    assert called == [1] and rt.capability_probe.calls == 0


def test_the_done_event_says_why_the_agent_was_skipped(client, monkeypatch):
    H, uid, kb, user, db, rt, conf = _seeded(client, "cap_sse", probe=StubProbe([UNSUPPORTED]))
    db.close()
    monkeypatch.setattr(conf, "agent_enabled", True)

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "问", "stream": True})
    done = sse_events(r.text)[-1]

    assert done["type"] == "done"
    assert "不支持工具调用" in done["agent_skipped"]
