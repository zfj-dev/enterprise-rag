"""认证：注册 / 登录（JWT）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.schemas import LoginRequest, RegisterRequest, TokenResponse
from app.models.entities import User
from app.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


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
    u = db.query(User).filter(User.username == body.username).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    return TokenResponse(access_token=create_access_token(u.username, u.role), role=u.role)
