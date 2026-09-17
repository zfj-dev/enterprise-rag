"""认证：注册 / 登录（JWT）。"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import get_settings
from app.core.schemas import LoginRequest, RegisterRequest, TokenResponse
from app.models.entities import User
from app.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# 登录限流(单 worker 假设,进程内滑动窗口)
_login_attempts: dict[str, list[float]] = {}
_login_lock = threading.Lock()
_LOGIN_WINDOW = 60.0
# 这张表的键是**攻击者可控的用户名**：不清就是个只涨不跌的内存泄漏
# （拿一万个随机用户名登录一万次就够了）。
_LOGIN_MAX_KEYS = 4096


def _login_allowed(username: str, limit: int) -> bool:
    """60 秒滑动窗口内允许 limit 次;超限返回 False。"""
    now = time.time()
    with _login_lock:
        ts = [t for t in _login_attempts.get(username, []) if now - t < _LOGIN_WINDOW]
        if len(ts) >= limit:
            _login_attempts[username] = ts
            return False
        ts.append(now)
        if len(_login_attempts) >= _LOGIN_MAX_KEYS and username not in _login_attempts:
            _prune_login_attempts(now)
        _login_attempts[username] = ts
        return True


def _prune_login_attempts(now: float) -> None:
    """先清过期的窗口；还满就按「最后尝试时间」丢掉最旧的一半。调用方须已持 `_login_lock`。"""
    for key in [k for k, v in _login_attempts.items()
                if not v or now - v[-1] >= _LOGIN_WINDOW]:
        _login_attempts.pop(key, None)
    if len(_login_attempts) >= _LOGIN_MAX_KEYS:
        stale = sorted(_login_attempts, key=lambda k: _login_attempts[k][-1])
        for key in stale[: _LOGIN_MAX_KEYS // 2]:
            _login_attempts.pop(key, None)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "用户名已存在")
    u = User(username=body.username, password_hash=hash_password(body.password), role="viewer")
    db.add(u)
    db.commit()
    db.refresh(u)
    return TokenResponse(access_token=create_access_token(u.username, u.role), role=u.role)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    if not _login_allowed(body.username, get_settings().login_rate_limit_per_min):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "尝试过于频繁,请稍后再试")
    u = db.query(User).filter(User.username == body.username).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    return TokenResponse(access_token=create_access_token(u.username, u.role), role=u.role)
