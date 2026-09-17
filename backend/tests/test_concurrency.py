"""并发与生命周期（#64 批 2）：共享可变状态在读写线程之间必须有一致的口径。

这些 bug 平时看不出来、压力下才现形，所以测试就照着那个交错来构造。
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

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


def test_indexing_is_undone_when_the_document_was_deleted_mid_flight(client, monkeypatch, tmp_path):
    """删除正好落在索引写入之后、状态提交之前 —— 撤不回来，删掉的文档就一直能被检索到。

    时序在 `_index_units` 里精确制造（它写完索引、返回之后才轮到那个检查），
    所以这条是确定性的，不靠线程碰运气。
    """
    from sqlalchemy import delete

    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.models.entities import Chunk, Document
    from app.services import document_service
    from tests.helpers import register_and_kb

    _, uid, kb = register_and_kb(client, "ghost_doc")
    rt = get_runtime()
    path = tmp_path / "幽灵.txt"
    path.write_text("绝密代号 GHOSTTOKEN9931 的营业收入。", encoding="utf-8")

    db = SessionLocal()
    try:
        doc = Document(kb_id=kb, owner_id=uid, filename="幽灵.txt",
                       file_path=str(path), status="processing")
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id

        real_index = document_service._index_units

        def index_then_delete(rt_, units, d):
            real_index(rt_, units, d)               # 先把索引写进去（真实顺序就是这样）
            other = SessionLocal()                  # 就在这一刻，另一个请求把文档删了
            try:
                other.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
                other.delete(other.get(Document, doc_id))
                other.commit()
            finally:
                other.close()

        monkeypatch.setattr(document_service, "_index_units", index_then_delete)
        document_service.process_document(db, rt, doc)      # 同步跑，确定性
    finally:
        db.close()

    assert rt.vector_store.search([0.0] * 1024, top_k=10,
                                  filter_meta={"doc_id": doc_id}) == []
    assert rt.bm25.search("绝密代号", top_k=10) == []
    follow = SessionLocal()
    try:
        assert follow.query(Chunk).filter(Chunk.doc_id == doc_id).count() == 0
    finally:
        follow.close()


def test_progress_is_not_reported_as_done_for_an_unknown_document():
    """内存表里没有 ≠ 做完了。重启前留下的 `processing` 曾经会被印成「100%」。"""
    from app.services.document_service import get_progress

    assert get_progress("从没见过的文档", "processing") == 0
    assert get_progress("从没见过的文档", "failed") == 0
    assert get_progress("从没见过的文档", "indexed") == 100


def test_the_progress_table_does_not_grow_forever():
    """只描述「本进程正在处理的」—— 终态要清掉，不然每个上传过的文档都留一条。"""
    from app.services.document_service import _PROGRESS, _set_progress

    _set_progress("tmp-doc-1", 35)
    assert _PROGRESS["tmp-doc-1"] == 35
    _set_progress("tmp-doc-1", 100)
    assert "tmp-doc-1" not in _PROGRESS


def test_the_capability_cache_does_not_remember_a_conservative_default():
    """保守默认是**内部降级值**、不是结论：落进缓存 = 一次 4xx 永久关掉某人的代理（#64）。"""
    from app.core.capability import CachedCapabilityProbe, ModelCapability, conservative_capability

    class _Probe:
        def __init__(self):
            self.calls = 0

        def probe_report(self, base_url, api_key, model):
            self.calls += 1
            return conservative_capability("探测被拒（HTTP 400）"), ""

    inner = _Probe()
    probe = CachedCapabilityProbe(inner)
    probe.probe_report("https://a.example.com/v1", "k", "m")
    probe.probe_report("https://a.example.com/v1", "k", "m")

    assert inner.calls == 2                     # 没被缓存，第二次还会再探
    assert probe.cached("https://a.example.com/v1", "k", "m") is None


def test_the_capability_cache_does_remember_a_probed_conclusion():
    from app.core.capability import CachedCapabilityProbe, ModelCapability

    class _Probe:
        def __init__(self):
            self.calls = 0

        def probe_report(self, base_url, api_key, model):
            self.calls += 1
            return ModelCapability(supports_tools=True, context_window=8192, source="probed"), ""

    inner = _Probe()
    probe = CachedCapabilityProbe(inner)
    probe.probe_report("https://b.example.com/v1", "k", "m")
    got, _ = probe.probe_report("https://b.example.com/v1", "k", "m")

    assert inner.calls == 1                     # 真结论要缓存（不然每次问答都探一遍）
    assert got.supports_tools is True


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


# ---------- 重启留下的残局（#64 批 3）----------

def test_a_document_left_processing_by_a_restart_is_marked_failed(client):
    """进程在处理到一半时重启：后台线程没了，文档就永远卡在 processing。

    它既不会被重跑（线程没了）、也不会被重建索引（`reindex_all` 只看 `indexed`），
    前端于是永远显示「处理中」—— 用户既等不到结果、也不知道该重传。
    """
    from app.db.session import SessionLocal
    from app.models.entities import Document
    from app.services.document_service import fail_stale_processing
    from tests.helpers import register_and_kb

    _, uid, kb = register_and_kb(client, "stale_processing")
    db = SessionLocal()
    try:
        stale = Document(kb_id=kb, owner_id=uid, filename="半截.pdf",
                         file_path="/nowhere/半截.pdf", status="processing")
        done = Document(kb_id=kb, owner_id=uid, filename="好的.pdf",
                        file_path="/nowhere/好的.pdf", status="indexed")
        db.add_all([stale, done])
        db.commit()

        assert fail_stale_processing(db) == 1
        db.refresh(stale)
        db.refresh(done)

        assert stale.status == "failed"
        assert "重启" in stale.error          # 原因要写出来，不能只丢一个状态
        assert done.status == "indexed"       # 已入库的一个也不许动
    finally:
        db.close()


# ---------- 客户端断开时的并发名额（#64 批 3）----------

def _drive_disconnect(resp, spec_version: str, check) -> None:
    """按 ASGI 驱动一次请求，并让客户端在**第一段字节之后**走人 —— 两条分支各走各的。

    - `spec_version < 2.4`：Starlette 起一个断连监听任务，靠 `receive` 收到
      `http.disconnect` 把流取消掉；
    - `>= 2.4`：没有那个任务，断连体现为「再往连接里写字节就抛 OSError」。

    收尾路径不同（后者连 `background` 都到不了），所以两条都得真的跑一遍。
    `check` 在响应**跑完但还没收掉悬着的异步生成器**时调用 —— 那正是名额该还回来的时刻，
    晚了就分不清是「这条路径还的」还是「生成器被回收时顺带还的」。
    """
    from starlette.requests import ClientDisconnect

    state = {"body": False, "chunks": 0, "gone": False}

    async def receive():
        if not state["body"]:
            state["body"] = True
            return {"type": "http.request", "body": b"", "more_body": False}
        while state["chunks"] < 1:
            await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            state["chunks"] += 1
            if spec_version != "2.3" and not state["gone"]:
                state["gone"] = True
                raise OSError("客户端已经走了：写不进去")

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/api/v1/chat/stream", "raw_path": b"/api/v1/chat/stream",
             "query_string": b"", "root_path": "", "headers": [],
             "client": ("test", 1), "server": ("test", 80)}
    loop = asyncio.new_event_loop()
    try:
        try:
            loop.run_until_complete(resp(scope, receive, send))
        except ClientDisconnect:
            assert spec_version != "2.3"     # 只有 ≥2.4 那条分支会把断连变成异常
        check()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())   # 到这儿才轮到「回收」那条路
        loop.close()


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_the_stream_slot_is_freed_as_soon_as_the_client_disconnects(client, monkeypatch, spec_version):
    """断开后名额要马上还回来 —— 只挂生成器的 `finally` 是不够的。

    Starlette 的线程池迭代被取消时**不会关闭**那个同步生成器，`finally` 要等它被回收才跑；
    表现出来就是「点了停止，几秒内再问就 429」。这里把响应对象捏在手里驱动一次断连：
    驱动完生成器仍然开着（下面断言了），名额却必须已经还回来。
    """
    from starlette.responses import StreamingResponse

    from app.api.deps import get_runtime
    from app.api.v1 import chat
    from app.db.session import SessionLocal
    from app.models.entities import User
    from tests.helpers import register_and_kb

    _, uid, kb = register_and_kb(client, "stream_disconnect")

    closed: list = []

    def fake_stream(*_a, **_k):
        """永不停歇的流：断连之后它还开着，`finally` 也就还没跑。"""
        def body():
            try:
                while True:
                    yield {"type": "delta", "text": "x"}
            finally:
                closed.append(True)
        return body()

    monkeypatch.setattr(chat, "stream_answer", fake_stream)
    db = SessionLocal()
    try:
        user = db.get(User, uid)
        resp = chat.chat_stream(body=chat.ChatRequest(kb_id=kb, question="问", stream=True),
                                user=user, db=db, rt=get_runtime())
    finally:
        db.close()
    assert isinstance(resp, StreamingResponse)

    guard = chat._stream_guard

    def _slots_are_all_back() -> None:
        tokens = []
        while len(tokens) <= guard.limit:   # 断连那一刻就该整份还回来，所以现在能占满
            t = guard.try_acquire(uid)
            if t is None:
                break
            tokens.append(t)
        for t in tokens:
            guard.release(uid, t)
        assert len(tokens) == guard.limit
        assert closed == []                 # 生成器还开着 —— 名额不是靠它被回收才回来的

    _drive_disconnect(resp, spec_version, _slots_are_all_back)
