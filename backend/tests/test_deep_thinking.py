"""「深度思考」接到代理链路（票 37 / #45）：全局开关表示「允许」，请求里的 deep 表示「这次用」。

两条纪律：
- **默认仍然是不用代理**：不发 deep 一律走确定性链路，哪怕部署方开了全局开关 ——
  spec 0002 的「默认关」在**请求层**同样成立。
- **全局开关是闸门不是默认值**：部署方没允许，请求里勾了也不能用。
"""
from __future__ import annotations

from app.services import chat_service
from tests.helpers import AGENT_RESULT as _AGENT_RESULT, seed_chat_doc, sse_events


def _done(r):
    return [e for e in sse_events(r.text) if e["type"] == "done"][0]


def _ask(client, H, kb, **extra):
    body = {"kb_id": kb, "question": "营收多少？", "stream": True}
    body.update(extra)
    return client.post("/api/v1/chat/stream", headers=H, json=body)


# ---------- 默认仍然不用代理 ----------

def test_omitting_deep_keeps_the_agent_off_even_when_the_deployment_allows_it(client, monkeypatch):
    """不发 deep = 不用代理。「默认关」不能只在服务层成立、到了请求层就变成默认开。"""
    called = []
    H, kb, user, db, rt = seed_chat_doc(client, "deep_absent")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
    monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: called.append(1))

    done = _done(_ask(client, H, kb))

    assert called == []                      # 代理那套一行都没跑
    assert not done.get("agent")             # 事件里如实说「没走代理」


def test_deep_false_is_the_same_as_omitting_it(client, monkeypatch):
    called = []
    H, kb, user, db, rt = seed_chat_doc(client, "deep_false")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
    monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: called.append(1))

    done = _done(_ask(client, H, kb, deep=False))

    assert called == [] and not done.get("agent")


# ---------- 全局闸门优先 ----------

def test_the_global_switch_is_a_gate_not_a_default(client, monkeypatch):
    """部署方没开代理，请求里勾了也不能用 —— 那是运维的许可，不是用户的选项。"""
    called = []
    H, kb, user, db, rt = seed_chat_doc(client, "deep_gate")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", False)
    monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: called.append(1))

    done = _done(_ask(client, H, kb, deep=True))

    assert called == [] and not done.get("agent")


# ---------- 两边都满足才走代理 ----------

def test_deep_plus_the_global_switch_routes_through_the_agent(client, monkeypatch):
    called = []
    H, kb, user, db, rt = seed_chat_doc(client, "deep_on")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
    monkeypatch.setattr(chat_service, "_run_agent_for",
                        lambda *a, **k: called.append(1) or dict(_AGENT_RESULT))

    r = _ask(client, H, kb, deep=True)
    done = _done(r)

    assert called == [1]
    assert done["answer"] == _AGENT_RESULT["answer"]
    assert done["agent"] is True             # 前端要能看出**这次确实走了代理**
    assert done["sources"] == _AGENT_RESULT["sources"]


def test_the_agent_flag_is_absent_from_the_deterministic_path(client, monkeypatch):
    """契约只增不改：确定性链路照旧，只是多了一个恒为假的 agent 字段。"""
    H, kb, user, db, rt = seed_chat_doc(client, "deep_contract")
    db.close()
    done = _done(_ask(client, H, kb))

    assert "agent" in done and not done["agent"]
    assert done["message_id"] and done["answer"]          # 既有字段一个不少


# ---------- 代理用不上时要如实说 ----------

def test_a_blocked_agent_degrades_and_says_why(client, monkeypatch):
    """自带模型不支持工具（票 34）→ 降级为单步，并说明原因；**不能**报成走了代理。"""
    H, kb, user, db, rt = seed_chat_doc(client, "deep_blocked")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
    monkeypatch.setattr(chat_service, "_agent_blocked_reason",
                        lambda rt_, prep_: "自带模型 m 不支持工具调用，本次降级为单步回答")

    done = _done(_ask(client, H, kb, deep=True))

    assert not done.get("agent")
    assert "降级" in (done.get("agent_skipped") or "")


# ---------- 前端要知道这个部署允不允许用代理 ----------

def test_health_tells_the_frontend_whether_the_agent_is_available(client, monkeypatch):
    """按钮该不该出现，由服务端说了算 —— 前端别自己猜。"""
    assert client.get("/health").json()["agent_enabled"] is False

    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)

    assert client.get("/health").json()["agent_enabled"] is True
