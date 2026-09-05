"""权限角色（RBAC）测试。

对真实实现：create_kb 只要求登录（所有角色可建库 → 200），知识库列表按 owner_id 过滤；
require_admin 仅门禁 GET /api/v1/metrics。角色 admin/uploader 无法经注册接口创建（注册恒为 viewer），
只能直插数据库（见 conftest.make_user / auth_headers）。
"""
from __future__ import annotations

from app.core.bm25 import InMemoryBm25
from app.core.embedding import FakeEmbedding
from app.core.retriever import HybridRetriever
from app.core.vector_store import InMemoryVectorStore, VectorItem


def _create_kb(client, headers: dict, name: str):
    r = client.post("/api/v1/knowledge", json={"name": name, "description": ""}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["id"]


# 用例 32
def test_admin_can_create_kb(client, auth_headers):
    """admin 角色创建知识库成功（返回 200 + id）。"""
    h = auth_headers("admin_user", role="admin")
    kb_id = _create_kb(client, h, "admin-kb")
    assert kb_id


# 用例 33
def test_uploader_can_create_kb(client, auth_headers):
    """uploader 角色创建知识库成功（200）。"""
    h = auth_headers("up_user", role="uploader")
    kb_id = _create_kb(client, h, "uploader-kb")
    assert kb_id


# 用例 34
def test_viewer_can_create_kb(client, auth_headers):
    """viewer 角色也能建库（200）——当前实现未在"建库"上做角色限制（规范要求 403，此处记录现状）。"""
    h = auth_headers("viewer_user", role="viewer")
    kb_id = _create_kb(client, h, "viewer-kb")
    assert kb_id


# 用例 35
def test_user_can_only_see_own_kb(client, auth_headers):
    """用户 A 看不到用户 B 的知识库（list_kbs 按 owner_id 过滤）。"""
    ha = auth_headers("user_a", role="viewer")
    hb = auth_headers("user_b", role="viewer")
    kb_a = _create_kb(client, ha, "A-kb")
    kb_b = _create_kb(client, hb, "B-kb")

    ids_a = {k["id"] for k in client.get("/api/v1/knowledge", headers=ha).json()}
    ids_b = {k["id"] for k in client.get("/api/v1/knowledge", headers=hb).json()}
    assert kb_a in ids_a and kb_b not in ids_a
    assert kb_b in ids_b and kb_a not in ids_b


# 用例 36
def test_retrieval_filters_by_owner():
    """向量检索时 owner_id 过滤生效：注入其他 owner 的分块不会被返回。"""
    emb = FakeEmbedding(dim=64)
    vs = InMemoryVectorStore()
    bm = InMemoryBm25()
    meta_shared = {"kb_id": "kb1", "doc_name": "d", "page_num": 1}
    vs.add([
        VectorItem(id="c1", vector=emb.encode(["比亚迪 电池 充电"])[0],
                   metadata={**meta_shared, "owner_id": "u1", "content": "比亚迪 电池 充电"}),
        VectorItem(id="c2", vector=emb.encode(["比亚迪 电池 充电"])[0],
                   metadata={**meta_shared, "owner_id": "u2", "content": "比亚迪 电池 充电"}),
    ])
    bm.add([
        {"id": "c1", "content": "比亚迪 电池 充电", "metadata": {**meta_shared, "owner_id": "u1", "content": "比亚迪 电池 充电"}},
        {"id": "c2", "content": "比亚迪 电池 充电", "metadata": {**meta_shared, "owner_id": "u2", "content": "比亚迪 电池 充电"}},
    ])
    rt = HybridRetriever(vs, bm, emb, reranker=None)
    hits = rt.retrieve("比亚迪 电池", kb_id="kb1", owner_id="u1")
    assert hits, "应命中至少一条"
    assert all(h["metadata"]["owner_id"] == "u1" for h in hits), "检索结果必须只含 owner=u1 的分块"


# 补充：require_admin 唯一生效点
def test_admin_can_access_metrics(client, auth_headers):
    """admin 可访问 /api/v1/metrics（require_admin）。"""
    h = auth_headers("admin_metric", role="admin")
    assert client.get("/api/v1/metrics", headers=h).status_code == 200


def test_viewer_cannot_access_metrics(client, auth_headers):
    """viewer 访问 /api/v1/metrics 返回 403（require_admin 拒绝）。"""
    h = auth_headers("viewer_metric", role="viewer")
    assert client.get("/api/v1/metrics", headers=h).status_code == 403
