"""文档上传 / 列表 / 状态。"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.core.container import Runtime
from app.core.schemas import DocumentOut
from app.models.entities import Document, KnowledgeBase, User
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_out(d: Document) -> DocumentOut:
    return DocumentOut(id=d.id, filename=d.filename, status=d.status,
                       page_count=d.page_count, chunk_count=d.chunk_count, error=d.error)


@router.post("", response_model=DocumentOut)
def upload(kb_id: str, file: UploadFile, user: User = Depends(get_current_user),
           db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    if not (db.get(KnowledgeBase, kb_id) and db.get(KnowledgeBase, kb_id).owner_id == user.id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权访问该知识库")
    os.makedirs("uploaded_files", exist_ok=True)
    dest = os.path.join("uploaded_files", file.filename)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    doc = Document(kb_id=kb_id, owner_id=user.id, filename=file.filename,
                   file_path=dest, status="pending")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    document_service.process_document(db, rt, doc)
    db.refresh(doc)
    return _to_out(doc)


@router.get("", response_model=list[DocumentOut])
def list_docs(kb_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.kb_id == kb_id, Document.owner_id == user.id).all()
    return [_to_out(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentOut)
def doc_status(doc_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    return _to_out(doc)
