"""认证：注册 / 登录（JWT）/ 改密 / 登出。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.api.deps import bearer_optional, get_current_user, get_db
from app.config import get_settings
from app.core.schemas import ChangePasswordIn, LoginRequest, RegisterRequest, TokenResponse
from app.models.entities import User
from app.utils.ratelimit import SlidingWindowLimiter
from app.utils.security import (create_access_token, decode_token, hash_password,
                                revoke_token, verify_password)

router = APIRouter(prefix="/auth", tags=["auth"])

# 登录限流（单 worker 假设，进程内滑动窗口，按**用户名**计）。
# 键是攻击者可控的，所以表必须封顶 —— 上限与清理策略都在 SlidingWindowLimiter 里。
_LOGIN_WINDOW = 60.0
_LOGIN_MAX_KEYS = 4096
_login_limiter = SlidingWindowLimiter(_LOGIN_WINDOW, max_keys=_LOGIN_MAX_KEYS)
# 底层计数表（测试与 conftest 按用户名清理它）
_login_attempts = _login_limiter.table

# 注册限流：窗口一小时、键是**客户端 IP**（还没登录，拿不到用户身份）。
_REGISTER_WINDOW = 3600.0
_register_limiter = SlidingWindowLimiter(_REGISTER_WINDOW, max_keys=_LOGIN_MAX_KEYS)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    # 按 IP 限流：这个接口不鉴权，不限流就是「开放建号」—— 每个号都能烧服务端的 LLM 额度
    # （安全审查 F5）。键只能是 IP（还没登录，没有用户身份可用）。
    ip = (request.client.host if request.client else "") or "?"
    if not _register_limiter.allow(ip, get_settings().register_rate_limit_per_hour):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "注册过于频繁,请稍后再试")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "用户名已存在")
    u = User(username=body.username, password_hash=hash_password(body.password), role="viewer")
    db.add(u)
    db.commit()
    db.refresh(u)
    return TokenResponse(access_token=create_access_token(u.username, u.role), role=u.role)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    if not _login_limiter.allow(body.username, get_settings().login_rate_limit_per_min):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "尝试过于频繁,请稍后再试")
    u = db.query(User).filter(User.username == body.username).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    return TokenResponse(access_token=create_access_token(u.username, u.role), role=u.role)


@router.post("/change-password")
def change_password(body: ChangePasswordIn, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """改自己的口令。

    这是**收回一个泄漏口令的唯一途径**：seed 出来的 `admin123` 是公开的，没有这个接口
    就只能去改数据库。旧口令必须对得上，所以拿到的令牌并不能直接改掉密码。
    """
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "原密码不正确")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"ok": True}


@router.post("/logout")
def logout(cred: HTTPAuthorizationCredentials | None = Depends(bearer_optional)):
    """把当前令牌拉黑至它自然过期。

    只用 `bearer_optional` 而不要求 `get_current_user`：用户行被删掉时也应该能登出。
    没有令牌、或令牌已过期/伪造 → 什么都不记（`decode_token` 过不了签名），直接返回 ok，
    所以这个接口即便不带凭证也不会成为一张只涨不跌的表。
    """
    payload = decode_token(cred.credentials) if cred else None
    if payload:
        revoke_token(payload.get("jti"), float(payload.get("exp") or 0))
    return {"ok": True}
