"""调试面板：查看某问题的检索中间结果（trace）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.core.container import Runtime
from app.core.schemas import DebugTraces
from app.models.entities import User
from app.services.chat_service import prepare

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/query", response_model=DebugTraces)
def debug_query(kb_id: str, question: str, user: User = Depends(get_current_user),
                db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    prep = prepare(db, rt, user, kb_id, question)
    return DebugTraces(query=prep.trace.get("query", question), rewrite=prep.trace.get("rewrite"),
                       retrieval_top=prep.trace.get("retrieval_top", []),
                       reranked=prep.candidates[:5], answer="")
