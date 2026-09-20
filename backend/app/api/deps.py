"""FastAPI 依赖：DB 会话 / Runtime 单例 / JWT 鉴权 / 角色。"""
from __future__ import annotations

import threading

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.container import Runtime, build_runtime
from app.db.session import get_db
from app.models.entities import KnowledgeBase, User
from app.utils.security import decode_token, is_revoked

# 对外可见（auth.py 的 /logout 直接用它取令牌）：不因缺凭证而 401，
# 拿不到就交给调用方自己决定怎么处理。
bearer_optional = HTTPBearer(auto_error=False)
_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    """懒加载单例。

    不变量：**只建一个**。lifespan 里那次 `build_runtime()` 失败会被 `_reindex` 的
    try/except 吞掉（`_runtime` 仍是 None），此后第一批并发请求会各建一个 Runtime ——
    各自一份向量库与嵌入模型，检索结果互相看不见。锁 + 双重检查关掉它。
    """
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_runtime()
    return _runtime


def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(bearer_optional),
    db: Session = Depends(get_db),
) -> User:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未提供凭证")
    payload = decode_token(cred.credentials)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "凭证无效/过期")
    # 登出过的令牌在自然过期前一律拒绝（否则「登出」只是前端清了个缓存）
    if is_revoked(payload.get("jti")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "凭证已登出")
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
    return user


def get_owned_kb(db: Session, kb_id: str, user: User) -> KnowledgeBase:
    """取知识库并**断言属主**；不属主一律 404（**不泄漏它存不存在**）。

    与 `chat_service._owned`（会话属主断言）对称：归属只该收口在一处，别让每个新入口
    再抄一遍 `if not kb or kb.owner_id != user.id` —— 抄漏一个就是越权（安全审查 F4；
    检索本身按 owner 过滤不会泄内容，但 kb_id 会进语义缓存的键、也会被拿去建会话）。
    """
    kb = db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "知识库不存在")
    return kb
