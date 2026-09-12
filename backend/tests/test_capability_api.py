"""能力/连通性回显接口（票 36 / #44）：把票 34 的探测结论**给界面看**。

两条纪律在这条链路上同样成立：
- **没探到就说没探到**，不把「不知道」渲染成「不支持工具」（那是在编答案）。
- **打开面板不发外网请求**：GET 只读缓存，探测由 POST 显式触发。
"""
from __future__ import annotations

from app.core.capability import CapabilityProbe, ModelCapability, conservative_capability
from tests.helpers import offline_resolver as _resolver, register_and_kb

PROBED_TOOLS_OK = ModelCapability(supports_tools=True, context_window=None, source="probed",
                                  note="探到 provider 接受了带 tools 的请求")


class StubCapability(CapabilityProbe):
    """脚本化的探测器：`reports` 是 (结论|None, 原因) 序列；`cached` 模拟已有缓存。"""

    def __init__(self, reports=(), cached=None):
        self._reports = list(reports)
        self._cached = cached
        self.calls = 0

    def probe(self, base_url, api_key, model):
        return self.probe_report(base_url, api_key, model)[0]

    def cached(self, base_url, api_key, model):
        return self._cached

    def probe_report(self, base_url, api_key, model):
        self.calls += 1
        return self._reports.pop(0) if self._reports else (None, "脚本用尽")


def _seeded(client, name, *, own_model=False, probe=None):
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
    if probe is not None:
        rt.capability_probe = probe
    return H, uid, kb, user, db, rt


# ---------- 未配置 / 地址不安全：都不该发起探测 ----------

def test_without_an_own_model_there_is_nothing_to_probe(client):
    stub = StubCapability([(PROBED_TOOLS_OK, "")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_none", probe=stub)
    try:
        body = client.get("/api/v1/llm/capability", headers=H).json()
    finally:
        db.close()

    assert body["configured"] is False and body["checked"] is False
    assert body["supports_tools"] is None                   # 不是 False：没配置 ≠ 不支持
    assert "未配置" in body["note"]
    assert stub.calls == 0                                  # 没配置就别去探


def test_an_unsafe_base_url_is_not_probed(client):
    """地址没过安全复查时**不发起探测** —— 探测本身也是一条对外发请求的路径（同票 33/35）。"""
    stub = StubCapability([(PROBED_TOOLS_OK, "")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_unsafe", own_model=True, probe=stub)

    def unsafe(base_url):
        return ["base_url 指向私网"]

    rt.url_resolver = unsafe
    try:
        body = client.get("/api/v1/llm/capability", headers=H).json()
        body2 = client.post("/api/v1/llm/capability/probe", headers=H).json()
    finally:
        db.close()

    # reachable 留 None：**没试过就不知道** —— 报「不通」等于编答案
    assert body["reachable"] is None and "安全复查" in body["note"]
    assert body2["reachable"] is None                       # POST 也不探
    assert stub.calls == 0


# ---------- GET 只读缓存，不发网络请求 ----------

def test_get_never_fires_a_probe(client):
    """打开面板不该打一次外网 —— GET 只看已有结论。"""
    stub = StubCapability([(PROBED_TOOLS_OK, "")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_get", own_model=True, probe=stub)
    try:
        body = client.get("/api/v1/llm/capability", headers=H).json()
    finally:
        db.close()

    assert body["checked"] is False and body["reachable"] is None
    # 不写「还没探测过」：探过但没探到同样没缓存，那句会变成假话
    assert "暂无结论" in body["note"]
    assert stub.calls == 0


def test_get_reports_an_existing_conclusion(client):
    stub = StubCapability(cached=PROBED_TOOLS_OK)
    H, uid, kb, user, db, rt = _seeded(client, "cap_hit", own_model=True, probe=stub)
    try:
        body = client.get("/api/v1/llm/capability", headers=H).json()
    finally:
        db.close()

    assert body["checked"] is True and body["reachable"] is True
    assert body["supports_tools"] is True and body["source"] == "probed"
    assert stub.calls == 0                                  # 命中缓存，不必再探


# ---------- POST 显式触发探测 ----------

def test_post_probes_and_reports_the_conclusion(client):
    stub = StubCapability([(PROBED_TOOLS_OK, "")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_post", own_model=True, probe=stub)
    try:
        body = client.post("/api/v1/llm/capability/probe", headers=H).json()
    finally:
        db.close()

    assert body["checked"] is True and body["reachable"] is True
    assert body["supports_tools"] is True and body["source"] == "probed"
    assert stub.calls == 1


def test_a_failed_probe_is_not_reported_as_unsupported(client):
    """**最关键的一条**：探不到时 supports_tools 必须是 null —— 不能谎报「不支持工具」。"""
    stub = StubCapability([(None, "连不上（ConnectError）")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_fail", own_model=True, probe=stub)
    try:
        body = client.post("/api/v1/llm/capability/probe", headers=H).json()
    finally:
        db.close()

    assert body["checked"] is True
    assert body["supports_tools"] is None
    assert body["reachable"] is None
    assert "连不上" in body["note"]                          # 原因要带出来，不能只说「失败」


def test_a_rejected_probe_still_counts_as_reachable(client):
    """被 4xx 拒了说明对端**答话了** —— 连通性为真，只是工具支持按保守默认。"""
    stub = StubCapability([(conservative_capability("探测被拒（HTTP 400）—— 无法据此确认工具支持，"
                                                    "按保守默认处理"), "")])
    H, uid, kb, user, db, rt = _seeded(client, "cap_400", own_model=True, probe=stub)
    try:
        body = client.post("/api/v1/llm/capability/probe", headers=H).json()
    finally:
        db.close()

    assert body["reachable"] is True
    assert body["supports_tools"] is False and body["source"] == "conservative"
    assert "无法据此确认" in body["note"]


# ---------- 真实探测器：把失败原因说清楚 ----------

def test_the_real_probe_explains_why_it_could_not_tell(monkeypatch):
    """401/404/429 与「支不支持工具」无关 —— 原因要按状态码分开写，不能笼统说一句失败。"""
    import httpx

    from app.core.capability import OpenAICompatCapabilityProbe

    class Resp:
        def __init__(self, code):
            self.status_code = code

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, *a, **k):
            return Resp(401)

    monkeypatch.setattr(httpx, "Client", Client)
    cap, reason = OpenAICompatCapabilityProbe(timeout=1).probe_report("https://a/v1", "k", "m")

    assert cap is None                                      # 没结论 → 不落缓存（票 34 的语义不能破）
    assert "401" in reason and "支不支持工具" in reason


def test_a_network_error_names_the_exception(monkeypatch):
    import httpx

    from app.core.capability import OpenAICompatCapabilityProbe

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, *a, **k):
            raise httpx.ConnectError("连不上")

    monkeypatch.setattr(httpx, "Client", Client)
    cap, reason = OpenAICompatCapabilityProbe(timeout=1).probe_report("https://a/v1", "k", "m")

    assert cap is None and "ConnectError" in reason


def test_probe_keeps_its_old_behaviour_for_the_agent_path(monkeypatch):
    """`probe()` 是代理降级的判据（票 34），加了诊断口径之后行为必须一模一样。"""
    import httpx

    from app.core.capability import OpenAICompatCapabilityProbe

    class Resp:
        status_code = 200

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(httpx, "Client", Client)
    cap = OpenAICompatCapabilityProbe(timeout=1).probe("https://a/v1", "k", "m")

    assert cap is not None and cap.supports_tools is True and cap.context_window is None


def test_the_explicit_test_button_reprobes_even_on_a_cache_hit(client):
    """「测试连通性」必须**真探一次**：用户刚改完地址或密钥，吃旧缓存等于没测。"""
    stub = StubCapability([(PROBED_TOOLS_OK, "")], cached=conservative_capability("旧结论：不确定"))
    H, uid, kb, user, db, rt = _seeded(client, "cap_fresh", own_model=True, probe=stub)
    try:
        got = client.post("/api/v1/llm/capability/probe", headers=H).json()
        # 只读的那条仍然吃缓存
        cached = client.get("/api/v1/llm/capability", headers=H).json()
    finally:
        db.close()

    assert stub.calls == 1                                  # 显式测试绕开缓存读
    assert got["source"] == "probed" and got["supports_tools"] is True
    assert cached["source"] == "conservative"               # GET 依旧只读缓存
