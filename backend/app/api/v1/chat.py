"""问答端点：SSE 流式（先 sources 事件，再增量 answer，最后 done）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.core.container import Runtime
from app.core.schemas import ChatRequest
from app.models.entities import User
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
