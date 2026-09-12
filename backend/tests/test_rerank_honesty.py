"""重排降级要看得见（票 39 / #48，C 案）。

重排挂掉时降级为 RRF 原顺序**是对的**（不该把回答也打断），但降级必须**能被看见** ——
否则一页评测报告会把「RRF 原顺序」的数字说成「重排已跑」，而这正是本仓库反复强调的
「拿不到就说拿不到」的反面。两条一起上：

- **B 可见**：降级时通过 `info` 出参把事实带出来，一路进 `timings` → trace → `done` 事件。
- **A 严格**：`RERANK_STRICT=true` 时**直接抛**，让评测照仓库惯例写「未跑」而不是出一个像模像样的数字。
"""
from __future__ import annotations

import httpx
import pytest

from app.core.bm25 import InMemoryBm25
from app.core.embedding import FakeEmbedding
from app.core.reranker import ApiReranker, Reranker, SiliconFlowReranker
from app.core.retriever import HybridRetriever
from app.core.vector_store import InMemoryVectorStore, VectorItem


def _stub_httpx(monkeypatch, *, raises=False, body=None):
    class Resp:
        status_code = 200

        def json(self):
            return body or {"results": []}

        def raise_for_status(self):
            pass

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            if raises:
                raise httpx.ConnectError("连不上")
            return Resp()

    monkeypatch.setattr(httpx, "Client", Client)


def _settings(monkeypatch, **kw):
    from app.config import get_settings

    s = get_settings()
    for k, v in kw.items():
        monkeypatch.setattr(s, k, v)
    return s


CANDIDATES = [{"chunk_id": "a", "content": "甲"}, {"chunk_id": "b", "content": "乙"}]


# ---------- A：严格模式下直接抛，不悄悄降级 ----------

@pytest.mark.parametrize("cls", [SiliconFlowReranker, ApiReranker])
def test_strict_mode_raises_instead_of_degrading(monkeypatch, cls):
    """评测要的是「这段没跑成」，不是一个把 RRF 顺序当成重排结果的数字。"""
    _stub_httpx(monkeypatch, raises=True)
    _settings(monkeypatch, rerank_strict=True, rerank_api_base="https://node.example.com")

    with pytest.raises(httpx.ConnectError):
        cls().rerank("q", CANDIDATES)


@pytest.mark.parametrize("cls", [SiliconFlowReranker, ApiReranker])
def test_default_mode_still_degrades_rather_than_breaking_the_answer(monkeypatch, cls):
    """线上默认仍是降级：重排挂掉不该把问答也打断。"""
    _stub_httpx(monkeypatch, raises=True)
    _settings(monkeypatch, rerank_strict=False, rerank_api_base="https://node.example.com")

    out = cls().rerank("q", CANDIDATES)

    assert [c["chunk_id"] for c in out] == ["a", "b"]      # RRF 原顺序


# ---------- B：降级要能通过 info 出参带出来 ----------

@pytest.mark.parametrize("cls", [SiliconFlowReranker, ApiReranker])
def test_a_degraded_rerank_reports_itself(monkeypatch, cls):
    _stub_httpx(monkeypatch, raises=True)
    _settings(monkeypatch, rerank_strict=False, rerank_api_base="https://node.example.com")
    info: dict = {}

    cls().rerank("q", CANDIDATES, info=info)

    assert info["degraded"] is True
    # 原因要带上，不能只说一句「失败了」——`or info["note"]` 那种写法恒真，等于没断言
    assert "连不上" in info["note"]
    assert "RRF" in info["note"]


@pytest.mark.parametrize("cls", [SiliconFlowReranker, ApiReranker])
def test_a_healthy_rerank_reports_no_degradation(monkeypatch, cls):
    _stub_httpx(monkeypatch, body={"results": [{"index": 0, "relevance_score": 0.9},
                                                {"index": 1, "relevance_score": 0.1}]})
    _settings(monkeypatch, rerank_strict=False, rerank_api_base="https://node.example.com")
    info: dict = {}

    cls().rerank("q", CANDIDATES, info=info)

    assert not info.get("degraded")


def test_partially_scored_results_are_flagged(monkeypatch):
    """「评分是 0」与「压根没被评分」是两回事 —— 后者是**不知道**，不该当成结论。"""
    _stub_httpx(monkeypatch, body={"results": [{"index": 0, "relevance_score": 0.9}]})   # 只回了一条
    _settings(monkeypatch, rerank_strict=False, rerank_api_base="https://node.example.com")
    info: dict = {}

    SiliconFlowReranker().rerank("q", CANDIDATES, info=info)

    assert info["unscored"] == 1


# ---------- 一路带到 timings / trace ----------

class _DegradingReranker(Reranker):
    """照着「托管重排不可达」的样子降级。"""

    def rerank(self, query, candidates, *, info=None):
        if info is not None:
            info["degraded"] = True
            info["note"] = "托管重排不可达，降级 RRF 顺序"
        return list(candidates)


def _retriever(reranker):
    emb = FakeEmbedding(dim=64)
    vs = InMemoryVectorStore()
    bm25 = InMemoryBm25()
    items = [VectorItem(id="c1", vector=emb.encode(["比亚迪 电池 手册"])[0],
                        metadata={"kb_id": "kb1", "owner_id": "u1", "doc_name": "d",
                                  "content": "比亚迪 电池 手册", "page_num": 1})]
    vs.add(items)
    bm25.add([{"id": "c1", "content": "比亚迪 电池 手册",
               "metadata": {"kb_id": "kb1", "owner_id": "u1", "content": "比亚迪 电池 手册"}}])
    return HybridRetriever(vs, bm25, emb, reranker=reranker)


def test_the_retriever_records_the_degradation_in_timings():
    """降级必须一路带出来 —— 只写日志等于只有看日志的人知道。"""
    timings: dict = {}
    _retriever(_DegradingReranker()).retrieve("比亚迪 电池", kb_id="kb1", owner_id="u1",
                                              timings=timings)

    assert timings["rerank_degraded"] is True
    assert "不可达" in timings["rerank_note"]


def test_a_healthy_rerank_leaves_no_degradation_mark():
    class _Fine(Reranker):
        def rerank(self, query, candidates, *, info=None):
            return list(candidates)

    timings: dict = {}
    _retriever(_Fine()).retrieve("比亚迪 电池", kb_id="kb1", owner_id="u1", timings=timings)

    assert timings["rerank_degraded"] is False
    assert not timings.get("rerank_note")


def test_the_trace_and_the_done_event_carry_the_degradation(client, monkeypatch):
    """要让**报告**看得见，而不是只有日志文件里一行 warning。"""
    from app.api.deps import get_runtime
    from app.services import chat_service
    from tests.helpers import sse_events, seed_chat_doc

    H, kb, user, db, rt = seed_chat_doc(client, "rerank_degraded")
    db.close()
    rt.retriever = _retriever(_DegradingReranker())

    prep = chat_service.prepare(db, rt, user, kb, "比亚迪 电池 手册")
    assert prep.trace["rerank_degraded"] is True
    assert "不可达" in prep.trace["rerank_note"]

    # 还得**真的下发到 done 事件**里 —— 只在 trace 里的话，读报告的人看不见
    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "比亚迪 电池 手册", "stream": True})
    done = [e for e in sse_events(r.text) if e["type"] == "done"][0]
    assert done["rerank"]["degraded"] is True
    assert "不可达" in done["rerank"]["note"]


def test_a_query_with_no_reranker_reports_unknown_not_false(client):
    """没有重排时 `degraded` 必须是 **null（不知道）**，不是 false（已断定没降级）。

    `bool(None)` 会把「不知道」写成「重排好好的」—— 正是仓库禁止的 null→false。
    """
    from app.services import chat_service
    from tests.helpers import sse_events, seed_chat_doc

    H, kb, user, db, rt = seed_chat_doc(client, "rerank_absent")
    db.close()
    rt.retriever = _retriever(None)          # 压根不重排

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "比亚迪 电池 手册", "stream": True})
    done = [e for e in sse_events(r.text) if e["type"] == "done"][0]

    assert done["rerank"]["degraded"] is None
