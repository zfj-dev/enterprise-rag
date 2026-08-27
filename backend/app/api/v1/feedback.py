"""用户反馈（点赞/点踩/纠错 → 落库，供评估闭环）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.schemas import FeedbackOut, FeedbackRequest
from app.models.entities import ChatMessage, Feedback, User

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut)
def submit(body: FeedbackRequest, user: User = Depends(get_current_user),
           db: Session = Depends(get_db)):
    if not db.get(ChatMessage, body.message_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "消息不存在")
    fb = Feedback(message_id=body.message_id, user_id=user.id, rating=body.rating, comment=body.comment)
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return FeedbackOut(id=fb.id, rating=fb.rating, comment=fb.comment)
