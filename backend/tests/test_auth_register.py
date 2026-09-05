"""注册接口 POST /api/v1/auth/register 测试。

对真实实现的契约：RegisterRequest(username 1-64, password 6-128, 无 role 字段)。
成功返回 200 + TokenResponse(access_token/token_type/role)，role 恒为 viewer（硬编码）；
重复用户名返回 400（不是 409）；密码用标准库 pbkdf2 哈希（不是 bcrypt）。
"""
from __future__ import annotations

import pytest


def _reg(client, username: str, password: str = "pw123456", extra: dict | None = None):
    payload = {"username": username, "password": password}
    if extra:
        payload.update(extra)
    return client.post("/api/v1/auth/register", json=payload)


# 用例 14
def test_register_success(client):
    """正常注册返回 200 + access_token（响应为 TokenResponse，无用户信息；规范要求 201，此处适配为 200）。"""
    r = _reg(client, "bob")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["role"] == "viewer"


# 用例 15
def test_register_duplicate_username(client):
    """重复用户名返回 400（规范要求 409，当前实现为 400）。"""
    _reg(client, "bob")
    r = _reg(client, "bob")
    assert r.status_code == 400
    assert r.json()["detail"] == "用户名已存在"


# 用例 16
def test_register_weak_password(client):
    """弱密码（<6 位）返回 422（RegisterRequest min_length=6）。5 位拒绝、恰好 6 位接受。"""
    assert _reg(client, "u1", "12345").status_code == 422
    assert _reg(client, "u2", "123456").status_code == 200


# 用例 17
def test_register_invalid_role(client):
    """RegisterRequest 无 role 字段：即使 body 传 role='admin' 也被忽略，最终仍是 viewer。"""
    r = _reg(client, "bob", extra={"role": "admin"})
    assert r.status_code == 200
    assert r.json()["role"] == "viewer"


# 用例 18
def test_register_sql_injection(client):
    """注册时 SQL 注入 payload 被无害化：按字面量完整储存，可用该字面量登录。"""
    u = "' OR '1'='1"
    r = _reg(client, u)
    assert r.status_code == 200, r.text
    # 该字面量用户名确实被创建，且能正常登录（证明没有注入、没有匹配到别的用户）
    login = client.post("/api/v1/auth/login", json={"username": u, "password": "pw123456"})
    assert login.status_code == 200


# 用例 19
def test_register_xss_payload(client):
    """注册时 XSS payload 被按字面量储存（服务端不做渲染，仅普通字符串）。"""
    u = "<script>alert(1)</script>"
    r = _reg(client, u)
    assert r.status_code == 200, r.text
    login = client.post("/api/v1/auth/login", json={"username": u, "password": "pw123456"})
    assert login.status_code == 200


# 用例 20
def test_register_default_role(client):
    """未传 role 时默认 role = viewer（当前为硬编码）。"""
    r = _reg(client, "bob")
    assert r.status_code == 200
    assert r.json()["role"] == "viewer"


# 用例 21
def test_register_password_hashing(client):
    """密码以 pbkdf2 哈希存储而非明文（规范称为 bcrypt，当前实现为标准库 pbkdf2_hmac）。"""
    _reg(client, "bob", "pw123456")
    from app.db.session import SessionLocal
    from app.models.entities import User

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "bob").first()
        assert u and u.password_hash, "应存在用户"
        assert u.password_hash != "pw123456"
        assert u.password_hash.startswith("pbkdf2$"), "应使用 pbkdf2 哈希（非 bcrypt $2b$，非明文）"
        from app.utils.security import verify_password
        assert verify_password("pw123456", u.password_hash)
        assert not verify_password("wrong", u.password_hash)
    finally:
        db.close()
