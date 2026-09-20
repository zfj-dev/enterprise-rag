"""密码哈希(标准库 pbkdf2) + JWT(PyJWT) + 令牌吊销，轻量无重依赖。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt

from app.config import get_settings

logger = logging.getLogger(__name__)

_ALGO_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ALGO_ITERATIONS)
    return f"pbkdf2${_ALGO_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_b64, dk_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception as e:
        logger.warning("verify_password 校验异常: %s", e)
        return False


# ---------- 令牌吊销（登出） ----------
# 键是 **jti**（令牌唯一 id），值是它的过期时刻。登出后到自然过期之间一律拒绝。
# 为什么要吊销：JWT 是自包含的，服务端不留档 —— 前端把 localStorage 一清，令牌在
# 服务端**照样有效**到 exp。没有这张表，「登出」就只是个视觉效果（安全审查 B1）。
#
# **已知边界**：表在进程内存里 —— 进程重启就没了，被吊销的令牌会「复活」到它自然过期。
# 与登录限流同一类边界。要跨进程/跨重启得挪到 Redis（配置里已有 redis_url）。
_REVOKED: dict[str, float] = {}
_revoked_lock = threading.Lock()


def _prune_revoked(now: float) -> None:
    """清掉已自然过期的条目 —— 过期后再拉黑没有意义，留着就是只涨不跌。调用方须持锁。"""
    for k in [k for k, exp in _REVOKED.items() if exp <= now]:
        _REVOKED.pop(k, None)


def revoke_token(jti: str | None, exp: float) -> None:
    """把令牌拉黑至其自然过期。jti 缺失（旧令牌）时什么都不做。"""
    if not jti:
        return
    with _revoked_lock:
        _REVOKED[jti] = exp
        _prune_revoked(time.time())


def is_revoked(jti: str | None) -> bool:
    if not jti:
        return False
    with _revoked_lock:
        return jti in _REVOKED


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    s = get_settings()
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or s.access_token_expire_minutes)
    # jti 是登出的抓手：没有它，`is_revoked` 无从判断该拉黑哪一个令牌。
    payload = {"sub": subject, "role": role, "exp": exp, "jti": uuid4().hex}
    return jwt.encode(payload, s.secret_key, algorithm=s.algorithm)


def decode_token(token: str) -> dict | None:
    s = get_settings()
    try:
        return jwt.decode(token, s.secret_key, algorithms=[s.algorithm])
    except Exception as e:
        logger.warning("decode_token 解析异常: %s", e)
        return None
