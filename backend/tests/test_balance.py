"""余额提醒（票 35 / #42）：能查则查，查不到就明说 —— 绝不拿我方用量冒充厂商余额。

三种结果（查到 / 厂商没这个接口 / 查失败）分得清清楚楚，任何一种都不会被说成「余额 0」。
"""
from __future__ import annotations

import httpx

from app.core.balance import (BalanceProbe, OpenAICompatBalanceProbe, VendorBalance,
                              balance_url, with_alert)
from tests.helpers import offline_resolver as _resolver, register_and_kb

DEEPSEEK_BODY = {"is_available": True, "balance_infos": [
    {"currency": "CNY", "total_balance": "110.00", "granted_balance": "10.00"}]}


class StubBalance(BalanceProbe):
    def __init__(self, result): self._result = result; self.calls = 0

    def fetch(self, base_url, api_key):
        self.calls += 1
        return self._result


def _stub_httpx(monkeypatch, *, status: int | None = None, body=None, raises: bool = False):
    class Resp:
        status_code = status

        def json(self): return body or {}

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, *a, **k):
            if raises:
                raise httpx.ConnectError("连不上")
            return Resp()

    monkeypatch.setattr(httpx, "Client", Client)


# ---------- 查到就展示真实余额 ----------

def test_the_balance_url_is_built_from_the_host_not_the_v1_base():
    """余额端点挂在**站根**：base_url 带 `/v1` 时直接拼会撞 404，被误报成「该厂商不支持」。"""
    assert balance_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/user/balance"
    assert balance_url("https://api.deepseek.com") == "https://api.deepseek.com/user/balance"
    assert balance_url("https://a.example.com/v1/") == "https://a.example.com/user/balance"


def test_a_non_json_200_is_reported_not_raised(monkeypatch):
    """对方回了个不是 JSON 的 200 —— 如实说「查不动」，而不是让接口 500 掉。"""
    class Resp:
        status_code = 200

        def json(self): raise ValueError("Expecting value: line 1 column 1")

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return Resp()

    monkeypatch.setattr(httpx, "Client", Client)

    got = OpenAICompatBalanceProbe().fetch("https://a/v1", "k")

    assert got.available is False and got.amount is None and "不是 JSON" in got.note


def test_a_vendor_that_exposes_balance_is_read_correctly(monkeypatch):
    _stub_httpx(monkeypatch, status=200, body=DEEPSEEK_BODY)

    got = OpenAICompatBalanceProbe().fetch("https://api.deepseek.com/v1", "k")

    assert got.available is True and got.amount == 110.0 and got.currency == "CNY"
    assert "厂商余额" in got.note and "deepseek" in got.note      # 这个数字从哪来，写清楚


def test_an_unrecognised_body_is_not_turned_into_a_number(monkeypatch):
    """返回体里没有可识别的余额字段 —— 说「没返回可识别的字段」，**不猜一个 0**。"""
    _stub_httpx(monkeypatch, status=200, body={"foo": "bar"})

    got = OpenAICompatBalanceProbe().fetch("https://a/v1", "k")

    assert got.available is False and got.amount is None and "可识别" in got.note


# ---------- 三种「没有数字」不许混成一句 ----------

def test_a_vendor_without_the_endpoint_says_so_explicitly(monkeypatch):
    """404 是**该厂商没有这个接口**，不是「余额 0」，也不是「查询失败」（票面第 2 条）。"""
    _stub_httpx(monkeypatch, status=404)

    got = OpenAICompatBalanceProbe().fetch("https://a/v1", "k")

    assert got.available is False and got.amount is None
    assert got.note == "该厂商不支持余额查询"


def test_a_rejected_query_is_not_reported_as_unsupported(monkeypatch):
    """401 是**凭据不对** —— 说成「该厂商不支持余额查询」就把人引偏了。"""
    _stub_httpx(monkeypatch, status=401)

    got = OpenAICompatBalanceProbe().fetch("https://a/v1", "k")

    assert "凭据不对" in got.note and "不支持" in got.note


def test_a_transient_failure_is_reported_as_a_failure(monkeypatch):
    _stub_httpx(monkeypatch, status=503)
    assert "查询失败" in OpenAICompatBalanceProbe().fetch("https://a/v1", "k").note

    _stub_httpx(monkeypatch, raises=True)
    assert "网络不通" in OpenAICompatBalanceProbe().fetch("https://a/v1", "k").note


# ---------- 提醒：只提醒，不拦截 ----------

def test_a_balance_below_the_threshold_raises_an_alert_that_does_not_block():
    got = with_alert(VendorBalance(available=True, amount=3.0, currency="CNY", note="厂商余额"),
                     threshold=10.0)

    assert got.available is True and got.amount == 3.0        # 余额照给
    assert "低于提醒线" in got.alert and "不拦截" in got.alert   # 只提醒


def test_a_balance_at_or_above_the_threshold_has_no_alert():
    assert with_alert(VendorBalance(available=True, amount=10.0, currency="CNY"), 10.0).alert == ""
    assert with_alert(VendorBalance(available=True, amount=99.0, currency="CNY"), 10.0).alert == ""


def test_an_unknown_balance_never_raises_an_alert():
    """查不到就说查不到 —— 不拿「未知」当「余额不足」去吓人。"""
    unknown = VendorBalance(available=False, note="该厂商不支持余额查询")

    assert with_alert(unknown, threshold=10.0).alert == ""
    assert with_alert(VendorBalance(available=True, amount=5.0), 0.0).alert == ""   # 没设提醒线


# ---------- 接进接口 ----------

def _seeded(client, name, *, probe=None, own_model=False):
    from app.api.deps import get_runtime
    from app.core.byok import InMemoryUserLLMConfigStore, LLMConfig
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = get_runtime()
    rt.url_resolver = _resolver()
    store = InMemoryUserLLMConfigStore()
    if own_model:
        store.set(uid, LLMConfig(base_url="https://api.example.com/v1", api_key="sk-own", model="m"))
    rt.user_llm_config_store = store
    rt.llm_factory = _FakeOwnFactory()      # 自带模型不真发请求：测试离线、确定性
    if probe is not None:
        rt.balance_probe = probe
    return H, uid, kb, user, db, rt


class _FakeOwnFactory:
    """把「自带模型」构造成假模型 —— 省掉真网络调用，本用例只关心余额与额度那条线。"""

    def build(self, cfg):
        from app.core.llm import FakeLLM

        return FakeLLM()


def test_without_their_own_model_there_is_no_vendor_balance_to_show(client):
    H, uid, kb, user, db, rt = _seeded(client, "bal_none")
    db.close()

    body = client.get("/api/v1/llm/balance", headers=H).json()

    assert body["vendor"]["available"] is False and body["vendor"]["amount"] is None
    assert "没有厂商余额可查" in body["vendor"]["note"]


def test_the_two_dimensions_are_reported_separately_with_their_sources(client):
    """厂商余额与我方统计用量**是两回事** —— 各自的来源要分别写明。"""
    H, uid, kb, user, db, rt = _seeded(
        client, "bal_two", own_model=True,
        probe=StubBalance(VendorBalance(available=True, amount=8.0, currency="CNY",
                                        note="厂商余额（api.example.com /user/balance）")))
    try:
        rt.usage_store.add(uid, {"model": "m", "input_tokens": 10, "output_tokens": 5,
                                 "source": "provider", "source_note": "", "cost": 0.25,
                                 "price_note": ""})
        body = client.get("/api/v1/llm/balance", headers=H).json()
    finally:
        db.close()

    assert body["vendor"]["amount"] == 8.0 and "厂商余额" in body["vendor"]["note"]
    assert body["ours"]["total_cost"] == 0.25
    assert "我方统计用量" in body["ours"]["note"]           # 不是余额，也写清楚了


def test_a_low_balance_only_warns_and_never_blocks(client, monkeypatch):
    from app.config import get_settings

    H, uid, kb, user, db, rt = _seeded(
        client, "bal_low", own_model=True,
        probe=StubBalance(VendorBalance(available=True, amount=1.0, currency="CNY", note="厂商余额")))
    try:
        monkeypatch.setattr(get_settings(), "byok_balance_alert_threshold", 10.0)
        body = client.get("/api/v1/llm/balance", headers=H).json()

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert "低于提醒线" in body["vendor"]["alert"]              # 措辞是「提醒」不是「拦截」
    assert r.status_code == 200                               # 只提醒，不拦截


def test_byok_usage_does_not_trigger_the_hard_block(client, monkeypatch):
    """回归（票面第 5 条）：自带 Key 花的是自己的钱 —— 用量照记，但**不纳入**平台硬拦。"""
    from app.config import get_settings

    H, uid, kb, user, db, rt = _seeded(client, "bal_quota", own_model=True)
    try:
        monkeypatch.setattr(get_settings(), "quota_enabled", True)
        monkeypatch.setattr(get_settings(), "quota_limit", 1.0)
        rt.usage_store.add(uid, {"model": "m", "input_tokens": 1, "output_tokens": 1,
                                 "source": "provider", "source_note": "", "cost": 99.0,
                                 "price_note": ""})           # 早就超了平台额度

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert r.status_code == 200                               # 不被硬拦
    assert len(rows) == 2                                     # 但用量照常记账（可审计）


def test_a_byok_config_that_fails_the_safety_recheck_is_not_exempt(client, monkeypatch):
    """地址复查不过时 llm_for 会回落到**服务端全局** —— 那时花的是我们的钱，就不能豁免。"""
    from app.config import get_settings

    H, uid, kb, user, db, rt = _seeded(client, "bal_quota_rebind", own_model=True)
    try:
        monkeypatch.setattr(get_settings(), "quota_enabled", True)
        monkeypatch.setattr(get_settings(), "quota_limit", 1.0)
        rt.usage_store.add(uid, {"model": "m", "input_tokens": 1, "output_tokens": 1,
                                 "source": "provider", "source_note": "", "cost": 99.0,
                                 "price_note": ""})
        rt.url_resolver = lambda host: ["10.0.0.7"]        # 域名被改指到内网 → 复查不过

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 402                            # 不能白送服务端额度


def test_a_user_without_their_own_model_is_still_blocked(client, monkeypatch):
    """反过来也要成立：没配自带模型的人照旧受平台额度约束（别把豁免放得太宽）。"""
    from app.config import get_settings

    H, uid, kb, user, db, rt = _seeded(client, "bal_quota_global")
    try:
        monkeypatch.setattr(get_settings(), "quota_enabled", True)
        monkeypatch.setattr(get_settings(), "quota_limit", 1.0)
        rt.usage_store.add(uid, {"model": "m", "input_tokens": 1, "output_tokens": 1,
                                 "source": "provider", "source_note": "", "cost": 99.0,
                                 "price_note": ""})

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 402
