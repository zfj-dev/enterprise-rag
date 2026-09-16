"""并发与生命周期（#64 批 2）：共享可变状态在读写线程之间必须有一致的口径。

这些 bug 平时看不出来、压力下才现形，所以测试就照着那个交错来构造。
"""
from __future__ import annotations

import threading

from app.core.bm25 import InMemoryBm25
from app.core.vector_store import InMemoryVectorStore, VectorItem


def _hammer(write, read, rounds=200):
    """一个写线程 + 一个读线程同时跑；返回读线程收集到的异常。"""
    stop = threading.Event()
    errs: list = []

    def writer():
        i = 0
        while not stop.is_set() and i < rounds:
            write(i)
            i += 1

    def reader():
        try:
            for _ in range(rounds):
                read()
        except Exception as e:      # noqa: BLE001 —— 这里就是要抓它
            errs.append(e)

    t1, t2 = threading.Thread(target=writer), threading.Thread(target=reader)
    t1.start(); t2.start(); t1.join()
    stop.set(); t2.join()
    return errs


def test_searching_while_indexing_does_not_explode():
    """上传在写、用户在问 —— 边遍历边改会 `RuntimeError`，于是上传期间提问直接 500。"""
    vs = InMemoryVectorStore()

    errs = _hammer(
        lambda i: vs.add([VectorItem(id="c%d" % i, vector=[0.1, 0.2], metadata={"kb_id": "k"})]),
        lambda: vs.search([0.1, 0.2], top_k=5))

    assert not errs, errs


def test_bm25_survives_concurrent_indexing():
    """`add` 是先写 `_docs` 再 `_rebuild`：中间那一刻语料与文档数对不上。"""
    bm = InMemoryBm25()

    errs = _hammer(
        lambda i: bm.add([{"id": "d%d" % i, "content": "文档 %d 比亚迪" % i,
                           "metadata": {"kb_id": "k"}}]),
        lambda: bm.search("比亚迪", top_k=3))

    assert not errs, errs


def test_bm25_scores_stay_attached_to_their_own_document():
    """分数按语料下标对齐 —— 用 `list(...).index(id)` 是 O(n²)，并发改动下还会贴错人。

    语料要够大：N=2、df=1 时 BM25 的 idf 恰好是 0，全部分数为 0，排序断言就没意义了。
    """
    bm = InMemoryBm25()
    bm.add([{"id": "a", "content": "比亚迪营业收入", "metadata": {"kb_id": "k"}}]
           + [{"id": "x%d" % i, "content": "第%d篇讲的是完全不相干的话题" % i,
               "metadata": {"kb_id": "k"}} for i in range(6)])

    hits = bm.search("比亚迪", top_k=3)

    assert hits[0]["chunk_id"] == "a"
    assert hits[0]["score"] > 0                 # 命中了就该有分，不是「排在最前但还是 0」


def test_the_runtime_singleton_is_built_only_once(monkeypatch):
    """lifespan 里那次建失败会被 `_reindex` 的 try/except 吞掉，此后并发请求会**各建一个**
    Runtime（各自一份向量库与嵌入模型，检索结果互相看不见）。"""
    import app.api.deps as deps

    built: list = []
    monkeypatch.setattr(deps, "_runtime", None)
    monkeypatch.setattr(deps, "build_runtime", lambda: built.append(1) or object())

    got: list = []
    threads = [threading.Thread(target=lambda: got.append(deps.get_runtime()))
               for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1                      # 只建了一次
    assert len({id(r) for r in got}) == 1       # 所有人拿到同一个


def test_a_racing_first_message_reuses_the_session_inserted_by_the_other_request(client):
    """前端拿本地 UUID 当会话 id：同一条会话的第一句话并发进来时，两边都「查不到 → 去插」，
    后插的撞 UNIQUE 直接 500。撞了就该改用已经插进去的那条。"""
    from app.db.session import SessionLocal
    from app.models.entities import ChatSession, User
    from app.services.chat_service import _get_or_create_session
    from tests.helpers import register_and_kb

    _, uid, kb = register_and_kb(client, "race_sess")
    sid = "race-session-1"

    other = SessionLocal()                       # 模拟「另一个请求」先把这条会话插进去
    try:
        other.add(ChatSession(id=sid, user_id=uid, kb_id=kb, title=""))
        other.commit()
    finally:
        other.close()

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uid).first()
        real_get, seen = db.get, {"n": 0}

        def fake_get(model, pk, *a, **k):
            # 本请求的第一次查询「恰好」发生在对方提交之前 —— 查不到，于是也去插
            if model is ChatSession and pk == sid and seen["n"] == 0:
                seen["n"] += 1
                return None
            return real_get(model, pk, *a, **k)

        db.get = fake_get
        sess = _get_or_create_session(db, user, kb, sid)

        assert sess is not None and sess.id == sid
    finally:
        db.close()
