"""文档入库管线：解析 → 按页分块 → 向量化 → 同时写关系库 + 向量库 + BM25 索引。

reindex_all: 容器重启后内存向量/BM25 索引清空，从数据库的已入库分块重建，
使"重启后仍能检索"，不用重新上传。
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.container import Runtime
from app.core.source_docs import make_meta
from app.core.vector_store import VectorItem
from app.models.entities import Chunk, Document

logger = logging.getLogger(__name__)

# 上传处理进度（0-100，内存态；重启后按 DB status 判断）
_PROGRESS: dict[str, int] = {}


def _set_progress(doc_id: str, val: int) -> None:
    """记进度。**终态不留档** —— 这张表只描述「本进程正在处理的」，
    不清理就是个只涨不跌的字典（每个上传过的文档都留一条）。"""
    if val >= 100 or val <= 0:
        _PROGRESS.pop(doc_id, None)
    else:
        _PROGRESS[doc_id] = val


def get_progress(doc_id: str, status: str | None = None) -> int:
    """处理进度。

    内存表里没有时**不许一律报 100**：那只能说明「这个进程没在处理它」——
    可能是重启前留下的 `processing`，此时报 100 会印出「处理中 100%」这种自相矛盾的东西。
    按文档状态给：indexed 才是做完，其余一律 0。
    """
    if doc_id in _PROGRESS:
        return _PROGRESS[doc_id]
    return 100 if status == "indexed" else 0


# 后台索引线程与「删除 / 覆盖文档」互斥。删除可能正好落在索引写入的中途：它删的时候
# 这些还没写进去，于是删完内容又被写回来 —— 已删的文档仍然能被检索到（#64 批 2）。
# 进门后各自**再确认一次文档还在不在**，光互斥是不够的。
DOC_WRITE_LOCK = threading.RLock()


def drop_document_file(db: Session, path: str | None) -> None:
    """删掉这个文档**独占**的文件。

    还有别的文档指向同一路径就不动它 —— 老库里的文档可能是共用一个路径存下来的
    （#64 之前按文件名全局存），删文件会把别人的文档一起弄坏。

    放在这一层（而不是某个路由里）：删文档与删知识库都要清盘，两边共用同一份
    「独占才删」的判断，免得抄一遍走样（安全审查 F2）。
    """
    if not path:
        return
    if db.query(Document).filter(Document.file_path == path).count():
        return
    try:
        os.remove(path)
    except OSError as e:      # noqa: BLE001 —— 清磁盘失败不该让删除接口失败
        logger.warning("删除文档文件失败(%s): %s", path, e)


# 上传文件的落盘约定：`<UPLOAD_ROOT>/<owner_id>/<kb_id>/<uuid>.<ext>`。
# 收口在这里，别让路由各写一份路径拼接（安全审查 F2）。
UPLOAD_ROOT = "uploaded_files"


def kb_upload_dir(owner_id: str, kb_id: str) -> str:
    return os.path.join(UPLOAD_ROOT, owner_id, kb_id)


def drop_kb_files(db: Session, owner_id: str, kb_id: str, paths: list[str]) -> None:
    """删一个知识库在磁盘上的全部文件。

    两步：先按 `Document.file_path` 逐个删（走上面「独占才删」的老逻辑，兼容 #64 之前
    那种全局共享路径）；再把这个库的目录整个清掉 —— 目录约定决定了底下只可能有这个库
    的文件，所以顺带带走**回滚 / 覆盖留下的孤儿**（安全审查 F2）。
    """
    for p in paths:
        drop_document_file(db, p)
    shutil.rmtree(kb_upload_dir(owner_id, kb_id), ignore_errors=True)


def _doc_still_exists(doc_id: str) -> bool:
    """用**独立 session** 问一句「这行还在吗」。

    不用处理线程自己那个 session：它手里那个 `doc` 还挂在 identity map 上，
    问不出真话（要么拿到旧对象，要么 ObjectDeletedError）。
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        return db.get(Document, doc_id) is not None
    finally:
        db.close()


def _abandon_deleted_document(rt: Runtime, doc_id: str) -> None:
    """处理途中文档被删了 —— 把我们已经写进去的**撤回**。

    删除接口跑的时候这些还没入库，所以只有这里来得及清。不清的话，删掉的文档
    会一直留在向量库 / BM25 / 关系库里被检索到，直到进程重启。
    """
    rt.vector_store.delete_by(doc_id=doc_id)
    rt.bm25.remove_by(doc_id=doc_id)
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        db.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        db.commit()
    except Exception as e:      # noqa: BLE001 —— 收拾残局失败只能记，不能把线程打死
        logger.warning("清理已删文档的残留失败(%s): %s", doc_id, e)
    finally:
        db.close()


def _index_units(rt: Runtime, child_units: list[dict], doc: Document) -> None:
    """把子块同时写入向量库 + BM25（含元数据：kb/owner/doc/page/content）。"""
    texts = [c["content"] for c in child_units]
    vectors = rt.embedding.encode(texts)
    items: list[VectorItem] = []
    bm25_entries: list[dict] = []
    for c, vec in zip(child_units, vectors):
        meta = make_meta(doc=doc, content=c["content"], page_num=c["page_num"])
        items.append(VectorItem(id=c["id"], vector=vec, metadata=meta))
        bm25_entries.append({"id": c["id"], "content": c["content"], "metadata": dict(meta)})
    rt.vector_store.add(items)
    rt.bm25.add(bm25_entries)


def process_document(db: Session, rt: Runtime, doc: Document) -> Document:
    try:
        doc.status = "processing"
        db.commit()
        _set_progress(doc.id, 5)
        if not os.path.exists(doc.file_path):
            raise FileNotFoundError(doc.file_path)
        _set_progress(doc.id, 15)

        parsed = rt.parser.parse(doc.file_path, doc.filename)
        _set_progress(doc.id, 35)
        if parsed.metadata.get("error"):
            raise ValueError(f"解析失败: {parsed.metadata['error']}")
        doc.page_count = parsed.page_count
        print(f"[doc] parser={parsed.metadata.get('parser', parsed.metadata.get('kind'))} "
              f"chars={len(parsed.text)} table={'|' in parsed.text}")

        page_texts = parsed.pages if parsed.pages else ([parsed.text] if parsed.text else [])
        all_chunks: list[dict] = []
        for pidx, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            all_chunks.extend(rt.chunker.chunk(page_text, doc_id=doc.id, page_num=pidx))

        child_units = [c for c in all_chunks if c["chunk_type"] == "child"]
        _set_progress(doc.id, 55)
        print(f"[doc] {doc.filename}: pages={doc.page_count} all_chunks={len(all_chunks)} child={len(child_units)}")

        # 写关系库
        db.add_all([
            Chunk(id=c["id"], parent_id=c["parent_id"], doc_id=doc.id, kb_id=doc.kb_id,
                  owner_id=doc.owner_id, content=c["content"], parent_content=c["parent_content"],
                  page_num=c["page_num"], chunk_type=c["chunk_type"])
            for c in all_chunks
        ])
        db.commit()

        import time as _t
        t0 = _t.time()
        _set_progress(doc.id, 70)
        _index_units(rt, child_units, doc)      # 乐观写入：写完之后再确认文档还在不在
        print(f"[doc] {doc.filename}: embed+index {len(child_units)} chunks in {_t.time()-t0:.1f}s")
        _set_progress(doc.id, 95)
        with DOC_WRITE_LOCK:
            if not _doc_still_exists(doc.id):
                # 处理途中文档被删了。撤回刚写进去的东西，**不要**再去 commit `doc`
                # （那行已经没了，commit 会抛 PendingRollbackError，而残留照样留在索引里）。
                _abandon_deleted_document(rt, doc.id)
                _set_progress(doc.id, 100)
                return doc
            doc.chunk_count = len(child_units)
            doc.status = "indexed"
            db.commit()
        _set_progress(doc.id, 100)
    except Exception as e:  # noqa
        print(f"[doc] {doc.filename} FAILED: {e}")
        try:
            db.rollback()
        except Exception as e:
            logger.warning("process_document 回滚失败: %s", e)
        _set_progress(doc.id, 0)
        try:
            d2 = db.get(Document, doc.id)
            if d2:
                _mark_failed(d2, str(e))
                db.commit()
        except Exception as e2:
            logger.warning("process_document 标记失败状态异常: %s", e2)
    return doc


def _mark_failed(doc: Document, reason: str) -> None:
    """标失败并把原因一起写上 —— 状态和原因成对出现，读的人才知道要不要重传。"""
    doc.status = "failed"
    doc.error = reason


def fail_stale_processing(db: Session) -> int:
    """把库里残留的 `processing` 标成 failed，返回改了几条 —— 启动时跑一次。

    残留只可能来自**上一次进程**：后台线程随进程一起没了，没人会再来收尾这些文档。
    而 `reindex_all` 只认 `indexed`，于是它们既检索不到、也不会被重跑，永远卡在「处理中」——
    用户既等不到结果，也不知道要重传。如实标成失败并写明原因，才是这个状态该有的样子。

    ⚠️ 这条假定**同一份库只有一个进程在写**（当前部署就是：`run_real.ps1` 与 Dockerfile 都没开
    `--workers`）。多 worker 时每个 worker 都会跑一遍启动收尾，后起的会把别人**正在索引**的
    文档标成失败；真要上多 worker，得改成按 worker 认领（例如加进程标识 / 心跳时间戳）。
    """
    rows = db.query(Document).filter(Document.status == "processing").all()
    for d in rows:
        _mark_failed(d, "服务重启时这份文档还在处理中，没能跑完 —— 请重新上传。")
    if rows:
        db.commit()
        print(f"[doc] {len(rows)} 份文档上次没处理完，已标为失败（等重传）")
    return len(rows)


def reindex_all(db: Session, rt: Runtime) -> int:
    """从数据库已入库的 child 分块重建内存向量库 + BM25 索引。返回重建的分块数。"""
    rows = (db.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.chunk_type == "child", Document.status == "indexed").all())
    if not rows:
        return 0
    chunks = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    texts = [c.content for c in chunks]
    vectors = rt.embedding.encode(texts)
    items: list[VectorItem] = []
    bm25_entries: list[dict] = []
    for c, d, vec in zip(chunks, docs, vectors):
        meta = {"kb_id": c.kb_id, "owner_id": c.owner_id, "doc_id": c.doc_id,
                "doc_name": d.filename, "page_num": c.page_num, "content": c.content}
        items.append(VectorItem(id=c.id, vector=vec, metadata=meta))
        bm25_entries.append({"id": c.id, "content": c.content, "metadata": dict(meta)})
    rt.vector_store.add(items)
    rt.bm25.add(bm25_entries)
    return len(chunks)


_processing_lock = threading.Lock()  # 串行处理：避免并发上传时 bm25/向量库/GPU 争用


def process_document_background(doc_id: str) -> None:
    """后台线程：用独立 session + 全局 runtime 单例处理一个文档。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal

    with _processing_lock:
        db: Session = SessionLocal()
        try:
            doc = db.get(Document, doc_id)
            if doc:
                process_document(db, get_runtime(), doc)
        finally:
            db.close()


# 在跑的后台索引线程：启动时登记、跑完就摘掉。关停时靠它等收尾（安全审查 F14）。
_live_threads: set[threading.Thread] = set()
_live_lock = threading.Lock()

SHUTDOWN_WAIT_SECONDS = 20.0


def _tracked(doc_id: str) -> None:
    try:
        process_document_background(doc_id)
    finally:
        with _live_lock:
            _live_threads.discard(threading.current_thread())


def launch_processing(doc_id: str) -> None:
    """上传接口调用：立刻返回，解析/嵌入在后台线程执行。"""
    t = threading.Thread(target=_tracked, args=(doc_id,), daemon=True)
    with _live_lock:
        _live_threads.add(t)
    t.start()


def wait_for_processing(timeout: float = SHUTDOWN_WAIT_SECONDS) -> int:
    """优雅关停：等在跑的后台索引收尾，返回**超时后仍在跑**的线程数（0 = 干净）。

    不等的话，正在索引的文档会被腰斩在「写了一半」的状态 —— 下次启动
    `fail_stale_processing` 会把它们标成 failed 让用户重传（那条兜底还在），
    但用户白传一次、也白解析一次。能等就等。
    """
    deadline = time.monotonic() + timeout
    while True:
        with _live_lock:
            alive = [t for t in _live_threads if t.is_alive()]
        if not alive:
            return 0
        left = deadline - time.monotonic()
        if left <= 0:
            return len(alive)
        alive[0].join(timeout=left)
