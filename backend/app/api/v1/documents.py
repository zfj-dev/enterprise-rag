"""文档上传 / 列表 / 状态 / 删除。上传异步处理：秒回 processing，后台解析+嵌入。"""
from __future__ import annotations

import logging
import os

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime
from app.config import get_settings
from app.core.container import Runtime
from app.core.parser import detect_kind
from app.core.schemas import DocumentOut
from app.models.entities import Chunk, Document, KnowledgeBase, User
from app.services import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_out(d: Document) -> DocumentOut:
    size = 0
    try:
        if d.file_path and os.path.exists(d.file_path):
            size = os.path.getsize(d.file_path)
    except Exception as e:
        logger.warning("读取文档文件大小失败(%s): %s", d.id, e)
        size = 0
    return DocumentOut(id=d.id, filename=d.filename, status=d.status,
                       page_count=d.page_count, chunk_count=d.chunk_count, error=d.error,
                       progress=document_service.get_progress(d.id), size=size,
                       created_at=d.created_at.isoformat() if d.created_at else "")


def _drop_file(db: Session, path: str | None) -> None:
    """删掉这个文档**独占**的文件。

    还有别的文档指向同一路径就不动它 —— 老库里的文档可能是共用一个路径存下来的
    （#64 之前按文件名全局存），删文件会把别人的文档一起弄坏。
    """
    if not path:
        return
    if db.query(Document).filter(Document.file_path == path).count():
        return
    try:
        os.remove(path)
    except OSError as e:      # noqa: BLE001 —— 清磁盘失败不该让删除接口失败
        logger.warning("删除文档文件失败(%s): %s", path, e)


@router.post("", response_model=DocumentOut)
def upload(kb_id: str, file: UploadFile, overwrite: bool = False,
           user: User = Depends(get_current_user), db: Session = Depends(get_db),
           rt: Runtime = Depends(get_runtime)):
    kb = db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权访问该知识库")

    # 文件名净化（防路径穿越）+ 类型白名单 + 大小限制
    raw_name = (file.filename or "").strip().replace("\\", "/")
    name = os.path.basename(raw_name)
    if (not name or name.startswith(".") or ".." in raw_name
            or "/" in raw_name or "\\" in raw_name):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "非法文件名")
    kind = detect_kind(name)
    if kind == "unknown":
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "不支持的文件类型（仅支持 PDF/Word/Excel/Markdown/TXT/图片）")
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = file.file.read()
    if max_bytes and len(content) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"文件超过 {get_settings().max_upload_mb}MB 上限")

    # 先判重名，**再**动盘上的文件 —— 反过来（老代码）会让「跳过重名」也把原文件覆盖掉。
    existing = (db.query(Document)
                .filter(Document.kb_id == kb_id, Document.owner_id == user.id,
                        Document.filename == name).first())
    if existing and not overwrite:
        return _to_out(existing)

    # 磁盘路径**不能只按文件名**：那是个全局命名空间，同名文件互相覆盖；而
    # /documents/{id}/file 是照着 file_path 把字节直接发出去的 —— 等于把别人的文件
    # 发给当前用户，后台线程还会拿被覆盖后的内容去索引（#64）。
    dest_dir = os.path.join("uploaded_files", user.id, kb_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, uuid4().hex + os.path.splitext(name)[1].lower())
    with open(dest, "wb") as f:
        f.write(content)

    if existing and overwrite:
        db.execute(delete(Chunk).where(Chunk.doc_id == existing.id))
        rt.vector_store.delete_by(doc_id=existing.id)
        rt.bm25.remove_by(doc_id=existing.id)
        old_path = existing.file_path
        db.delete(existing)
        db.commit()
        _drop_file(db, old_path)

    doc = Document(kb_id=kb_id, owner_id=user.id, filename=name,
                   file_path=dest, status="processing")
    try:
        db.add(doc)
        db.commit()
    except Exception:
        db.rollback()
        # 行没落库，刚写的文件就没人认领了 —— 留着就是永远没人清的垃圾
        # （清理逻辑是照着 Document.file_path 做的，没有行就没有入口）。
        _drop_file(db, dest)
        raise
    db.refresh(doc)
    document_service.launch_processing(doc.id)  # 后台异步解析+嵌入
    return _to_out(doc)


@router.get("", response_model=list[DocumentOut])
def list_docs(kb_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = (db.query(Document)
            .filter(Document.kb_id == kb_id, Document.owner_id == user.id)
            .order_by(Document.created_at.desc()).all())
    return [_to_out(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentOut)
def doc_status(doc_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    return _to_out(doc)


@router.get("/{doc_id}/content")
def doc_content(doc_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """按页返回文档文本（由 parent chunk 重建），供前端"点击来源-定位"用。"""
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    rows = (db.query(Chunk).filter(Chunk.doc_id == doc_id, Chunk.chunk_type == "parent")
            .order_by(Chunk.page_num.asc(), Chunk.id.asc()).all())
    pages: dict[int, list[str]] = {}
    for c in rows:
        pages.setdefault(c.page_num, []).append(c.content or "")
    return {"filename": doc.filename,
            "pages": [{"page": p, "text": "\n".join(v)} for p, v in sorted(pages.items())]}


@router.get("/{doc_id}/file")
def doc_file(doc_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from fastapi.responses import FileResponse
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文件不存在")
    media = "application/pdf" if doc.filename.lower().endswith(".pdf") else "application/octet-stream"
    return FileResponse(doc.file_path, media_type=media, filename=doc.filename)


@router.patch("/{doc_id}", response_model=DocumentOut)
def rename_doc(doc_id: str, body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    name = (body.get("filename") or "").strip()
    if name:
        doc.filename = name
        db.commit()
        db.refresh(doc)
    return _to_out(doc)


@router.delete("/{doc_id}")
def delete_doc(doc_id: str, user: User = Depends(get_current_user),
               db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    doc = db.get(Document, doc_id)
    if not doc or doc.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    db.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
    rt.vector_store.delete_by(doc_id=doc_id)
    rt.bm25.remove_by(doc_id=doc_id)
    path = doc.file_path
    db.delete(doc)
    db.commit()
    _drop_file(db, path)
    return {"ok": True}
