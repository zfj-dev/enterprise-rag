"""用户反馈（点赞/点踩/纠错 → 落库，供评估闭环）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.schemas import FeedbackOut, FeedbackRequest
from app.models.entities import ChatMessage, ChatSession, Feedback, User

router = APIRouter(prefix="/feedback", tags=["feedback"])

_RATINGS = (1, -1)


@router.post("", response_model=FeedbackOut)
def submit(body: FeedbackRequest, user: User = Depends(get_current_user),
           db: Session = Depends(get_db)):
    if body.rating not in _RATINGS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "rating 只能是 1（赞）或 -1（踩）")
    # 只能给自己会话里的消息打分。只查「消息存在」是不够的 —— 那样谁都能给别人的消息
    # 写反馈，而 feedback 是评测闭环的输入（#64）。不属主一律 404，不泄漏消息存不存在。
    own = (db.query(ChatMessage)
           .join(ChatSession, ChatMessage.session_id == ChatSession.id)
           .filter(ChatMessage.id == body.message_id, ChatSession.user_id == user.id)
           .first())
    if not own:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "消息不存在")
    fb = Feedback(message_id=body.message_id, user_id=user.id, rating=body.rating, comment=body.comment)
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return FeedbackOut(id=fb.id, rating=fb.rating, comment=fb.comment)
