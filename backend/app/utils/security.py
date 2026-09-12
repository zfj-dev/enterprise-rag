"""密码哈希(标准库 pbkdf2) + JWT(PyJWT)，轻量无重依赖。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone

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


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    s = get_settings()
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or s.access_token_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": exp}
    return jwt.encode(payload, s.secret_key, algorithm=s.algorithm)


def decode_token(token: str) -> dict | None:
    s = get_settings()
    try:
        return jwt.decode(token, s.secret_key, algorithms=[s.algorithm])
    except Exception as e:
        logger.warning("decode_token 解析异常: %s", e)
        return None
