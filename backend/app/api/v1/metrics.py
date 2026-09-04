"""评估指标（RAGAS 占位 + 反馈计数）。真实评估管线接 RAGAS 后填充。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.schemas import MetricsOut
from app.models.entities import ChatMessage, Feedback, User

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsOut)
def metrics(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    answered = db.query(ChatMessage).filter(ChatMessage.role == "assistant").count()
    return MetricsOut(total_answered=answered)
