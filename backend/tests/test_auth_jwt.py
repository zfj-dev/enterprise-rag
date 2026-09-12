"""JWT 安全测试。

对真实实现：PyJWT + secret_key/HS256；decode_token 默认验证 exp/iat/nbf 等（若存在），
但**不强制** exp/sub（缺 sub 时 get_current_user 会因 payload["sub"] KeyError 而 500，故本文件对
缺 sub 只测 decode_token 层面的当前行为）。受保护端点用 GET /api/v1/knowledge（require 登录）。
"""
from __future__ import annotations

import time

import jwt
import pytest

from app.config import get_settings
from app.utils.security import create_access_token, decode_token

EP = "/api/v1/knowledge"  # 需登录的受保护端点


def _now_ts() -> int:
    return int(time.time())


# 用例 22
def test_jwt_token_expired(client):
    """已过期 token（exp 在过去）访问受保护接口返回 401。"""
    tok = create_access_token("alice", "viewer", expires_minutes=-1)  # exp 在过去
    assert decode_token(tok) is None
    r = client.get(EP, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


# 用例 23
def test_jwt_token_future_iat(client):
    """未来时间签发的 token（iat 在未来）被拒绝：返回 401。"""
    secret = get_settings().secret_key
    tok = jwt.encode({"sub": "alice", "role": "viewer",
                      "iat": _now_ts() + 3600, "exp": _now_ts() + 7200}, secret, algorithm="HS256")
    assert decode_token(tok) is None
    r = client.get(EP, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


# 用例 24
def test_jwt_token_tampered_payload(client):
    """篡改 token 后访问返回 401（签名校验失败）。"""
    tok = create_access_token("alice", "viewer")
    tampered = tok[: len(tok) - 5] + "zzzzz"
    assert decode_token(tampered) is None
    r = client.get(EP, headers={"Authorization": f"Bearer {tampered}"})
    assert r.status_code == 401


# 用例 25
def test_jwt_token_none_algorithm(client):
    """alg=none 攻击：算法白名单为 HS256，none 被拒绝 → 401。"""
    tok = jwt.encode({"sub": "alice", "role": "viewer"}, "", algorithm="none")
    assert decode_token(tok) is None
    r = client.get(EP, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


# 用例 26
def test_jwt_token_missing_exp(client):
    """缺少 exp claim 的 token 当前不会被拒绝（decode 成功）——记录为安全缺口。"""
    secret = get_settings().secret_key
    tok = jwt.encode({"sub": "alice", "role": "viewer"}, secret, algorithm="HS256")
    payload = decode_token(tok)
    assert payload is not None, "当前实现未强制 exp，应能解码"
    assert payload["sub"] == "alice"


# 用例 27
def test_jwt_token_missing_sub(client):
    """缺少 sub claim 的 token 当前可解码（decode 层面），但受保护端点会因 KeyError 报 500——记录为缺口。"""
    secret = get_settings().secret_key
    tok = jwt.encode({"role": "viewer", "exp": _now_ts() + 3600}, secret, algorithm="HS256")
    payload = decode_token(tok)
    assert payload is not None and "sub" not in payload, "当前实现解码出含 role/exp 但无 sub 的 payload"


# 用例 28
def test_jwt_protected_endpoint_no_token(client):
    """无 Token 访问受保护接口返回 401。"""
    r = client.get(EP)
    assert r.status_code == 401


# 用例 29
def test_jwt_protected_endpoint_invalid_token(client):
    """传 'Bearer fake.token.here' 返回 401。"""
    r = client.get(EP, headers={"Authorization": "Bearer fake.token.here"})
    assert r.status_code == 401


# 用例 30
def test_jwt_refresh_flow(client):
    """当前未实现 refresh token 机制，测试整体跳过（记录为未实现功能）。"""
    pytest.skip("当前实现仅签发 access_token，无 refresh token / 刷新流程")


# 用例 31
def test_jwt_logout_invalidation(client):
    """当前未实现登出/黑名单机制（无 /auth/logout 端点），测试整体跳过（记录为未实现功能）。"""
    pytest.skip("当前实现无登出接口、无 token 黑名单/Redis 失效机制")
