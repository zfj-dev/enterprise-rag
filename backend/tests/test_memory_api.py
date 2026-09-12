"""记忆的列出与删除（票 25 / #32）。

用户能看到系统记住了自己什么、能删掉某一条，删除**立即生效**（下一问不再召回）。
越权删别人的记忆要被拒绝。
"""
from __future__ import annotations

from app.core.memory import InMemoryMemoryStore
from tests.helpers import StubEmbedding, register_and_kb

FACT = "用户在跟进电池项目"
QUERY = "电池进展如何"


def _wire(store):
    """装上带 stub 记忆缝的 Runtime —— 接口与召回都从它取 store / embedding。"""
    import app.api.deps as deps
    from app.core.container import build_runtime

    rt = build_runtime()
    rt.memory_store = store
    rt.embedding = StubEmbedding()
    deps._runtime = rt
    return rt


def _prepare(rt, uid, kb_id, question):
    from app.db.session import SessionLocal
    from app.models.entities import User
    from app.services import chat_service

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uid).first()
        return chat_service.prepare(db, rt, user, kb_id, question)
    finally:
        db.close()


# ---------- 列出 ----------

def test_list_returns_only_own_memories(client):
    H, uid, _kb = register_and_kb(client, "mlist")
    store = InMemoryMemoryStore()
    store.add(uid, [FACT])
    store.add("someone-else", ["别人的记忆"])
    _wire(store)

    r = client.get("/api/v1/memory", headers=H)
    assert r.status_code == 200
    assert [m["content"] for m in r.json()] == [FACT]


def test_list_empty_when_nothing_remembered(client):
    H, _uid, _kb = register_and_kb(client, "mempty")
    _wire(InMemoryMemoryStore())
    assert client.get("/api/v1/memory", headers=H).json() == []


def test_list_requires_auth(client):
    _wire(InMemoryMemoryStore())
    assert client.get("/api/v1/memory").status_code == 401


# ---------- 删除 ----------

def test_delete_removes_own_memory(client):
    H, uid, _kb = register_and_kb(client, "mdel")
    store = InMemoryMemoryStore()
    store.add(uid, [FACT, "用户偏好中文"])
    _wire(store)
    fid = store.list(uid)[0]["id"]

    assert client.delete("/api/v1/memory/{}".format(fid), headers=H).status_code == 200
    assert [m["content"] for m in client.get("/api/v1/memory", headers=H).json()] == ["用户偏好中文"]


def test_delete_unknown_memory_is_404(client):
    H, _uid, _kb = register_and_kb(client, "m404")
    _wire(InMemoryMemoryStore())
    assert client.delete("/api/v1/memory/nope", headers=H).status_code == 404


def test_cannot_delete_someone_elses_memory(client):
    """越权删除被拒绝，且对方那条记忆原封不动。"""
    H1, uid1, _kb1 = register_and_kb(client, "mmine")
    _H2, uid2, _kb2 = register_and_kb(client, "mother")
    store = InMemoryMemoryStore()
    store.add(uid1, [FACT])
    store.add(uid2, ["别人的记忆"])
    _wire(store)

    fid = store.list(uid2)[0]["id"]
    assert client.delete("/api/v1/memory/{}".format(fid), headers=H1).status_code == 404
    assert [m["content"] for m in store.list(uid2)] == ["别人的记忆"]


# ---------- 删除立即生效（下一问不再召回） ----------

def test_delete_takes_effect_on_the_next_recall(client):
    H, uid, kb = register_and_kb(client, "mrecall")
    store = InMemoryMemoryStore()
    store.add(uid, [FACT])
    rt = _wire(store)

    assert "【已知信息】" in _prepare(rt, uid, kb, QUERY).prompt      # 删之前召回到

    fid = store.list(uid)[0]["id"]
    assert client.delete("/api/v1/memory/{}".format(fid), headers=H).status_code == 200

    prep = _prepare(rt, uid, kb, QUERY)
    assert "【已知信息】" not in prep.prompt                          # 删之后同一问题不再召回
    assert prep.trace["memory_recalled"] == 0
