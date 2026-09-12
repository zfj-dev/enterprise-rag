"""PgVectorStore 集成测试（需真实 postgres + pgvector 扩展）。

CI 不设 TEST_PG_URL 时整体跳过；真机/生产验证时:
    TEST_PG_URL=postgresql+psycopg://user:pass@host:5432/db pytest tests/test_pgvector.py -v
"""
import os

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_PG_URL = os.environ.get("TEST_PG_URL")

pytestmark = pytest.mark.skipif(not TEST_PG_URL, reason="需要 TEST_PG_URL 指向真实 pgvector 库")

from app.core.vector_store import PgVectorStore, VectorItem  # noqa: E402
from app.models.entities import Base, Chunk, Document, KnowledgeBase, User  # noqa: E402


def test_search_filter_roundtrip():
    engine = create_engine(TEST_PG_URL, future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    vs = PgVectorStore(TEST_PG_URL, dim=4)

    with Session() as db:
        db.query(Chunk).delete()
        db.query(Document).delete()
        db.query(KnowledgeBase).delete()
        db.query(User).delete()
        db.commit()
        u = User(username="pguser", password_hash="x", role="viewer")
        db.add(u)
        db.commit()
        db.refresh(u)
        kb = KnowledgeBase(owner_id=u.id, name="kb")
        db.add(kb)
        db.commit()
        db.refresh(kb)
        d = Document(kb_id=kb.id, owner_id=u.id, filename="doc.txt", file_path="", status="indexed")
        db.add(d)
        db.commit()
        db.refresh(d)
        db.add_all([
            Chunk(id="pgc1", doc_id=d.id, kb_id=kb.id, owner_id=u.id,
                  content="比亚迪 电池 充电", page_num=1, chunk_type="child"),
            Chunk(id="pgc2", doc_id=d.id, kb_id=kb.id, owner_id=u.id,
                  content="华为 报销 流程", page_num=2, chunk_type="child"),
        ])
        db.commit()
        u_id, kb_id, d_id = u.id, kb.id, d.id

    # 已落库的分块行上 upsert embedding（add 对已存在行只更新 embedding，不触发 NOT NULL 缺失）
    vs.add([
        VectorItem(id="pgc1", vector=[1.0, 0, 0, 0], metadata={"doc_id": d_id, "kb_id": kb_id, "owner_id": u_id}),
        VectorItem(id="pgc2", vector=[0.0, 1.0, 0, 0], metadata={"doc_id": d_id, "kb_id": kb_id, "owner_id": u_id}),
    ])

    hits = vs.search([1.0, 0, 0, 0], top_k=5, filter_meta={"kb_id": kb_id, "owner_id": u_id})
    assert hits, "应至少命中一条"
    assert hits[0].id == "pgc1"
    assert hits[0].score > 0.9
    assert hits[0].metadata["doc_name"] == "doc.txt"
    assert hits[0].metadata["page_num"] == 1

    # 权限/库过滤：另一个 kb 应无结果
    no_hits = vs.search([1.0, 0, 0, 0], top_k=5, filter_meta={"kb_id": "nokb", "owner_id": u_id})
    assert no_hits == []

    # delete_by: 按 doc 删除后检索为空（doc_id/kb_id 各可选）
    vs.delete_by(doc_id=d_id)
    assert vs.search([1.0, 0, 0, 0], top_k=5, filter_meta={"kb_id": kb_id, "owner_id": u_id}) == []
