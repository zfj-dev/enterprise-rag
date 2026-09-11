"""问答端点：SSE 流式（先 sources 事件，再增量 answer，最后 done）。"""
from __future__ import annotations

import json
import threading

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.config import get_settings
from app.core.quota import check_quota
from app.core.container import Runtime
from app.core.schemas import ChatRequest
from app.models.entities import ChatMessage, ChatSession, User
from app.services.chat_service import prepare, stream_answer

router = APIRouter(prefix="/chat", tags=["chat"])


class _StreamGuard:
    """每用户同时流式对话计数守卫。limit=每用户上限；超限 try_acquire 返回 False。"""

    def __init__(self, limit: int):
        self.limit = limit
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        with self._lock:
            n = self._counts.get(key, 0)
            if n >= self.limit:
                return False
            self._counts[key] = n + 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            n = self._counts.get(key, 1)
            if n <= 1:
                self._counts.pop(key, None)
            else:
                self._counts[key] = n - 1


_stream_guard = _StreamGuard(get_settings().max_concurrent_streams_per_user)


@router.post("/stream")
def chat_stream(body: ChatRequest, user: User = Depends(get_current_user),
                db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    if body.session_id:
        s = db.get(ChatSession, body.session_id)
        if s and s.user_id != user.id:
            from fastapi import HTTPException
            raise HTTPException(404, "会话不存在")

    # 预算硬拦（票 29）：判定在**生成之前**、依据此前已累计的用量 —— 所以拦截只对**下一个**
    # 请求生效，已在进行中的流不被打断；管理员豁免（用量照记）。
    conf = get_settings()
    # 自带 Key 的用户不纳入硬拦：花的是他自己的钱（票 35）。
    # 但**必须是他自己的 key 真的在用**：地址复查不过时 llm_for 会回落到服务端全局，
    # 那时花的是我们的钱，就不能豁免（否则白送额度）。
    own = rt.user_llm_config_store.get(user.id)
    byok = own is not None and rt.base_url_is_still_safe(own.base_url)
    reason = check_quota(rt.usage_store, user, enabled=conf.quota_enabled,
                          window=conf.quota_window, limit=conf.quota_limit, byok=byok)
    if reason:
        from fastapi import HTTPException
        # 402：额度用尽（既不是「请求太快」，也不是「没权限」）
        raise HTTPException(402, reason)

    prep = prepare(db, rt, user, body.kb_id, body.question, body.session_id)

    if not _stream_guard.try_acquire(user.id):
        from fastapi import HTTPException
        raise HTTPException(429, "并发对话过多，请稍后再试")

    def gen():
        try:
            for ev in stream_answer(db, rt, prep):
                # 事件里可能含 date 之外字段，直接序列化整条
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _stream_guard.release(user.id)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/sessions")
def list_sessions(kb_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """列出该知识库的会话（标题/时间/最后消息预览/消息数）。"""
    sessions = (db.query(ChatSession)
                .filter(ChatSession.kb_id == kb_id, ChatSession.user_id == user.id)
                .order_by(ChatSession.created_at.desc())
                .all())
    out = []
    for s in sessions:
        last = (db.query(ChatMessage).filter(ChatMessage.session_id == s.id)
                .order_by(ChatMessage.created_at.desc()).first())
        cnt = (db.query(ChatMessage).filter(ChatMessage.session_id == s.id).count())
        out.append({
            "id": s.id,
            "title": s.title or "新对话",
            "created_at": s.created_at.isoformat() if s.created_at else "",
            "last_preview": (last.content or "")[:60] if last else "",
            "message_count": cnt,
        })
    return out


@router.get("/sessions/{session_id}")
def get_session(session_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """取单个会话的全部消息。"""
    s = db.get(ChatSession, session_id)
    if not s or s.user_id != user.id:
        from fastapi import HTTPException
        raise HTTPException(404, "会话不存在")
    msgs = (db.query(ChatMessage).filter(ChatMessage.session_id == s.id)
            .order_by(ChatMessage.created_at.asc()).all())
    return {"session_id": s.id, "title": s.title or "",
            "messages": [{"role": m.role, "content": m.content} for m in msgs]}


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """重命名会话标题。"""
    from fastapi import HTTPException
    s = db.get(ChatSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "会话不存在")
    title = (body.get("title") or "").strip()[:50]
    if title:
        s.title = title
        db.commit()
    return {"ok": True, "title": s.title}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """删除会话及其消息。"""
    from fastapi import HTTPException
    s = db.get(ChatSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "会话不存在")
    db.query(ChatMessage).filter(ChatMessage.session_id == s.id).delete()
    db.delete(s)
    db.commit()
    return {"ok": True}


@router.get("/history")
def chat_history(kb_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """取该知识库最近会话及消息，供刷新页面后恢复对话。"""
    sess = (db.query(ChatSession)
            .filter(ChatSession.kb_id == kb_id, ChatSession.user_id == user.id)
            .order_by(ChatSession.created_at.desc()).first())
    if not sess:
        return {"session_id": None, "messages": []}
    msgs = (db.query(ChatMessage)
            .filter(ChatMessage.session_id == sess.id)
            .order_by(ChatMessage.created_at.asc()).all())
    return {"session_id": sess.id,
            "messages": [{"role": m.role, "content": m.content} for m in msgs]}
