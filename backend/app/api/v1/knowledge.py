"""知识库管理（含删除：连带其文档、分块与检索索引）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.core.container import Runtime
from app.core.schemas import KnowledgeBaseCreate, KnowledgeBaseOut
from app.models.entities import Chunk, Document, KnowledgeBase, User

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_out(kb: KnowledgeBase, db: Session) -> KnowledgeBaseOut:
    return KnowledgeBaseOut(id=kb.id, name=kb.name, description=kb.description,
                            embedding_model=kb.embedding_model, doc_count=len(kb.documents))


@router.get("", response_model=list[KnowledgeBaseOut])
def list_kbs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return [_to_out(k, db) for k in db.query(KnowledgeBase).filter(KnowledgeBase.owner_id == user.id).all()]


@router.post("", response_model=KnowledgeBaseOut)
def create_kb(body: KnowledgeBaseCreate, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    kb = KnowledgeBase(owner_id=user.id, name=body.name, description=body.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _to_out(kb, db)


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, user: User = Depends(get_current_user),
              db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    kb = db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "知识库不存在")
    docs = db.query(Document).filter(Document.kb_id == kb_id).all()
    for doc in docs:
        db.execute(delete(Chunk).where(Chunk.doc_id == doc.id))
    rt.vector_store.delete_by(kb_id=kb_id)
    rt.bm25.remove_by(kb_id=kb_id)
    for doc in docs:
        db.delete(doc)
    db.delete(kb)
    db.commit()
    return {"ok": True, "deleted_docs": len(docs)}
