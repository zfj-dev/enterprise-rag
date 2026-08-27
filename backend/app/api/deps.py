"""FastAPI 依赖：DB 会话 / Runtime 单例 / JWT 鉴权 / 角色。"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.container import Runtime, build_runtime
from app.db.session import get_db
from app.models.entities import User
from app.utils.security import decode_token

_bearer = HTTPBearer(auto_error=False)
_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = build_runtime()
    return _runtime


def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未提供凭证")
    payload = decode_token(cred.credentials)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "凭证无效/过期")
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
    return user
