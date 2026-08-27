"""知识库管理。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.schemas import KnowledgeBaseCreate, KnowledgeBaseOut
from app.models.entities import KnowledgeBase, User

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_out(kb: KnowledgeBase, db: Session) -> KnowledgeBaseOut:
    count = len(kb.documents)
    return KnowledgeBaseOut(id=kb.id, name=kb.name, description=kb.description,
                            embedding_model=kb.embedding_model, doc_count=count)


@router.get("", response_model=list[KnowledgeBaseOut])
def list_kbs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.owner_id == user.id).all()
    return [_to_out(k, db) for k in kbs]


@router.post("", response_model=KnowledgeBaseOut)
def create_kb(body: KnowledgeBaseCreate, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    kb = KnowledgeBase(owner_id=user.id, name=body.name, description=body.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _to_out(kb, db)
