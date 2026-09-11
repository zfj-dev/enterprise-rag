"""跨会话记忆：列出与删除**自己**的记忆（票 25）。

存储层按 user 下推过滤，所以这里取不到、也删不掉别人的记忆。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user, get_runtime
from app.core.container import Runtime
from app.core.schemas import MemoryFactOut
from app.models.entities import User

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=list[MemoryFactOut])
def list_memories(user: User = Depends(get_current_user), rt: Runtime = Depends(get_runtime)):
    """列出当前用户的记忆。"""
    return rt.memory_store.list(user.id)


@router.delete("/{fact_id}")
def delete_memory(fact_id: str, user: User = Depends(get_current_user),
                  rt: Runtime = Depends(get_runtime)):
    """删除自己的一条记忆。不存在、或不是自己的，一律 404（不泄漏他人记忆的存在性）。"""
    if not rt.memory_store.delete(user.id, fact_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "记忆不存在")
    return {"ok": True}
