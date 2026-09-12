"""评测脚本共用的「本地起一套可问答的库」+ 几个人人要写一遍的小工具。

评测要在真实管线里跑，就得有：一个发起人、一个知识库、一份**真走过入库管线**的文档
（解析 -> 分块 -> 嵌入 -> 向量库 + BM25，还要落关系库 —— 枚举/编号检索要查 Chunk 表）。
落盘报告与「临时扳全局开关」也一样，三个 runner 共用这里的一份，别各写各的。
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


def write_report(path: str, lines: list[str]) -> None:
    """把报告落盘并打印路径 —— 所有评测脚本共用这一处。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(path)


def with_setting(ask, name: str, value):
    """把这次问答的某个**全局开关**临时扳到 value，问完立刻还原。

    两条配置共用同一套管线，只差这一个开关，差异归因才干净。`get_settings()` 是**进程级单例**，
    所以用它的多条链路只能**顺序跑**（`eval_compare.run_links` 正是顺序的）；并发跑会串台，
    那时得把开关做成显式参数而不是全局态。
    """
    from app.config import get_settings

    def wrapped(question: str):
        s = get_settings()
        was = getattr(s, name)
        setattr(s, name, value)
        try:
            return ask(question)
        finally:
            setattr(s, name, was)

    return wrapped


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
