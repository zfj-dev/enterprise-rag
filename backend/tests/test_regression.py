# -*- coding: utf-8 -*-
"""Step5 回归测试防线：契约 + 核心流。

约定：
- 通过类测试：验证"应当正确"的行为（覆盖核心链路），当前应 PASS。
- xfail(strict=True) 契约测试：把已知缺陷固化成"期望正确、当前错误"。
  现在标 xfail（不破坏套件）；谁修复后会出现 XPASS(strict) -> 套件失败 -> 提醒移除标记。
"""
from __future__ import annotations
import pytest


def _reg(client, name):
    r = client.post("/api/v1/auth/register", json={"username": name, "password": "pw123456"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _kb(client, H):
    r = client.post("/api/v1/knowledge", headers=H, json={"name": "回归库", "description": ""})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------- 通过类：核心链路 ----------
def test_health(client):
    assert client.get("/api/v1/auth/health").status_code == 200


def test_core_flow(client):
    H = _reg(client, "coreflow")
    kb = _kb(client, H)
    # 流式问答（空库也应有回复）
    with client.stream("POST", "/api/v1/chat/stream", headers=H,
                       json={"kb_id": kb, "question": "你好"}) as r:
        assert r.status_code == 200


def test_other_user_cannot_read_session(client):
    """越权：他人不能读/用我的会话（已修复，应 404）。"""
    a = _reg(client, "usera")
    b = _reg(client, "userb")
    kb = _kb(client, a)
    sess = None
    import re
    with client.stream("POST", "/api/v1/chat/stream", headers=a,
                       json={"kb_id": kb, "question": "越权", "session_id": "sess-abc"}) as r:
        for line in r.iter_lines():
            m = re.search(r'"session_id":\s*"([^"]+)"', line)
            if m:
                sess = m.group(1)
                break
    assert sess
    assert client.get(f"/api/v1/chat/sessions/{sess}", headers=b).status_code == 404


def test_upload_requires_kb_owner(client):
    a = _reg(client, "ownera")
    kb = _kb(client, a)
    b = _reg(client, "ownerb")
    r = client.post(f"/api/v1/documents?kb_id={kb}", headers=b,
                    files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 403  # 别人库应被拒


# ---------- xfail 契约：已知缺陷 ----------
# 已修复：LoginRequest 增加 min_length 校验
def test_register_empty_rejected(client):
    r = client.post("/api/v1/auth/register", json={"username": "", "password": ""})
    assert r.status_code in (400, 422)  # 期望拒绝；现在 200


# 已修复：filename 净化 + detect_kind 白名单 + 大小校验
def test_upload_type_traversal_rejected(client):
    H = _reg(client, "upx")
    kb = _kb(client, H)
    # 非白名单扩展名 + 路径穿越文件名
    r = client.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                    files={"file": ("../evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400  # 期望拒绝；现在 200 且写出目录


# 已修复：/metrics 已接 require_admin，公开泄露关闭
def test_admin_only_endpoint_enforced(client):
    # 普通 viewer 访问需 admin 的接口应 403；当前无任何 admin 校验
    H = _reg(client, "viewerx")
    # metrics 是示例接口，若某接口声明 admin-only 则此处应 403；当前全部放行
    r = client.get("/api/v1/metrics", headers=H)
    assert r.status_code != 200  # 期望受限；当前 200


# 已修复：上传接口读取 max_upload_mb，超限返回 413
def test_upload_size_limit_enforced(client, monkeypatch):
    from app.api.v1 import documents as doc_mod
    class _S:
        max_upload_mb = 0.0001  # ~104 bytes
    monkeypatch.setattr(doc_mod, "get_settings", lambda: _S())
    H = _reg(client, "sizex")
    kb = _kb(client, H)
    r = client.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                    files={"file": ("a.txt", b"x" * 200, "text/plain")})
    assert r.status_code == 413
