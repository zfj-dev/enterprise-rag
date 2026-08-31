"""问答端点：SSE 流式（先 sources 事件，再增量 answer，最后 done）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.core.container import Runtime
from app.core.schemas import ChatRequest
from app.models.entities import ChatMessage, ChatSession, User
from app.services.chat_service import prepare, stream_answer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
def chat_stream(body: ChatRequest, user: User = Depends(get_current_user),
                db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    prep = prepare(db, rt, user, body.kb_id, body.question, body.session_id)

    def gen():
        for ev in stream_answer(db, rt, prep):
            # 事件里可能含 date 之外字段，直接序列化整条
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

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
