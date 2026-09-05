"""登录接口 POST /api/v1/auth/login 测试。

对真实实现的契约：JSON body(LoginRequest: username/password 均 min_length=1)——
不是 OAuth2PasswordRequestForm；成功返回 200+access_token，失败返回 401（不区分用户不存在/密码错）。
"""
from __future__ import annotations

import time

import pytest


def _reg(client, username: str, password: str = "pw123456") -> None:
    r = client.post("/api/v1/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, r.text


def _login(client, username: str, password: str):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


# 用例 1
def test_login_success_valid_credentials(client):
    """正确用户名密码返回 200 + access_token。"""
    _reg(client, "alice")
    r = _login(client, "alice", "pw123456")
    assert r.status_code == 200
    body = r.json()
    assert body["access_token"]
    assert body["role"] == "viewer"


# 用例 2
def test_login_wrong_password(client):
    """错误密码返回 401，detail 用统一文案，不暴露"用户名不存在"还是"密码错误"。"""
    _reg(client, "alice")
    r = _login(client, "alice", "wrong091")
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 3
def test_login_nonexistent_user(client):
    """不存在的用户返回 401，且与错误密码的文案一致（防用户名枚举）。"""
    r = _login(client, "nobody", "pw123456")
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 4
def test_login_empty_username(client):
    """空用户名返回 422（LoginRequest min_length=1）。"""
    r = _login(client, "", "pw123456")
    assert r.status_code == 422


# 用例 5
def test_login_empty_password(client):
    """空密码返回 422（LoginRequest min_length=1）。"""
    r = _login(client, "alice", "")
    assert r.status_code == 422


# 用例 6
def test_login_sql_injection_attempt(client):
    """username 输入 SQL 注入 payload 返回 401（ORM 参数化查询安全处理，不会注入）。"""
    r = _login(client, "' OR '1'='1", "pw123456")
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 7
def test_login_xss_attempt(client):
    """username 输入 <script> 返回 401，服务端不执行脚本（仅作普通字符串校验）。"""
    r = _login(client, "<script>alert(1)</script>", "pw123456")
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 8
def test_login_super_long_username(client):
    """超长用户名（>256 字符）当前不触发 422（LoginRequest 无 max_length），因用户不存在而返回 401。"""
    r = _login(client, "x" * 300, "pw123456")
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 9
def test_login_super_long_password(client):
    """超长密码（>256 字符）当前不触发 422（LoginRequest 无 max_length），密码不匹配而返回 401。"""
    r = _login(client, "alice", "y" * 300)
    assert r.status_code == 401
    assert r.json()["detail"] == "用户名或密码错误"


# 用例 10
def test_login_unicode_username(client):
    """中文/emoji/特殊 Unicode 用户名可正常注册并登录。"""
    uname = "用户😀"
    _reg(client, uname)
    r = _login(client, uname, "pw123456")
    assert r.status_code == 200
    assert r.json()["access_token"]


# 用例 11
def test_login_case_sensitivity(client):
    """用户名大小写敏感：'Alice' 注册后，'alice' 登录失败、'Alice' 成功（sqlite 二进制排序实现）。"""
    _reg(client, "Alice")
    assert _login(client, "alice", "pw123456").status_code == 401
    assert _login(client, "Alice", "pw123456").status_code == 200


# 用例 12
def test_login_response_time(client):
    """登录接口单次响应时间 < 500ms（time 模块计时，未装 pytest-timeout）。"""
    _reg(client, "alice")
    t0 = time.perf_counter()
    for _ in range(5):
        assert _login(client, "alice", "pw123456").status_code == 200
    elapsed = (time.perf_counter() - t0) / 5
    assert elapsed < 0.5, f"登录平均耗时 {elapsed:.3f}s"


# 用例 13
def test_login_rate_limit(client):
    """快速连续登录 10 次：当前实现未做速率限制，应全部返回 401 而非 429（记录缺口）。"""
    _reg(client, "alice")
    codes = [_login(client, "alice", "wrong091").status_code for _ in range(10)]
    assert all(c == 401 for c in codes), f"当前不应触发 429，实际: {codes}"
