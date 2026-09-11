"""评测脚本共用的「本地起一套可问答的库」—— 别每个脚本各写一遍（同 app/eval_index.py 的用意）。

评测要在真实管线里跑，就得有：一个发起人、一个知识库、一份**真走过入库管线**的文档
（解析 -> 分块 -> 嵌入 -> 向量库 + BM25，还要落关系库 —— 枚举/编号检索要查 Chunk 表）。
"""
from __future__ import annotations

import os


def ensure_schema() -> None:
    """库表没建过就先建（幂等），**并给升级过列的老库补列** —— 评测脚本要能独立跑起来。

    `create_all` 只建表、不会给已有的表加列：少了补列这一步，老库上跑评测会当场撞
    `no such column`（与 app 启动的 lifespan 走同一处，见 app/db/migrate.py）。
    """
    from app.db.migrate import ensure_sqlite_columns
    from app.db.session import engine
    from app.models.entities import Base

    Base.metadata.create_all(bind=engine)
    ensure_sqlite_columns(engine)


def eval_user(db, username: str):
    """本地库里的固定评测用户（幂等）。"""
    from app.models.entities import User
    from app.utils.security import hash_password

    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username, password_hash=hash_password("eval-only"),
                    role="viewer")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def new_kb(db, owner_id: str, name: str):
    """建一个干净的知识库 —— 每次评测用新的，免得吃到上一轮的索引。"""
    from app.models.entities import KnowledgeBase

    kb = KnowledgeBase(owner_id=owner_id, name=name, description="评测用（脚本创建的）")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def drop_kb(db, kb_id: str) -> None:
    """删掉评测建的知识库（连带文档与分块）—— 反复跑不该在本地库里堆垃圾。"""
    from app.models.entities import Chunk, Document, KnowledgeBase

    db.query(Chunk).filter(Chunk.kb_id == kb_id).delete()
    db.query(Document).filter(Document.kb_id == kb_id).delete()
    db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).delete()
    db.commit()


def ingest_file(db, rt, path: str, owner_id: str, kb_id: str) -> str:
    """把磁盘上的一份文档走**真实入库管线**建进库，返回 doc_id。

    解析失败等异常原样抛出 —— 评测宁可当场报错，也不拿半份索引出数字。
    """
    from app.models.entities import Document
    from app.services.document_service import process_document

    doc = Document(kb_id=kb_id, owner_id=owner_id,
                   filename=os.path.basename(path), file_path=os.path.abspath(path))
    db.add(doc)
    db.commit()
    db.refresh(doc)
    process_document(db, rt, doc)
    db.refresh(doc)
    if doc.status != "indexed":
        raise RuntimeError("文档未入库（status=%s）：%s" % (doc.status, doc.error or path))
    return doc.id
