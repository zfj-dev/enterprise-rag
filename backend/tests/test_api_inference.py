"""ApiEmbedding / ApiReranker 单测：monkeypatch 替换 httpx.Client, 验证与推理节点的 HTTP 契约。

不引第三方 mock(如 respx),直接 mock httpx.Client;不联网。
"""
import httpx
import pytest

from app.core.embedding import ApiEmbedding
from app.core.reranker import ApiReranker


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class _FakeClient:
    """记录 post; /embed 返回与 texts 等长的向量, /rerank 返回 index 递减的 relevance_score。"""
    instances = []

    def __init__(self, *a, **k):
        self.posts = []
        _FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
        if url.endswith("/embed"):
            n = len((json or {}).get("texts", []))
            return _FakeResp({"vectors": [[0.0, 1.0] for _ in range(n)]})
        if url.endswith("/rerank"):
            docs = (json or {}).get("documents", [])
            return _FakeResp({"results": [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(len(docs))]})
        return _FakeResp({})


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient.instances.clear()


def test_api_embedding_batches_and_shapes():
    e = ApiEmbedding(base="http://node:9000", api_key="tok", batch_size=2)
    vecs = e.encode(["a", "b", "c"])   # 3 texts, batch_size=2 -> 2 次请求
    assert len(vecs) == 3
    assert all(len(v) == 2 for v in vecs)
    assert len(_FakeClient.instances) == 1          # 每次 encode 一个 Client
    assert len(_FakeClient.instances[0].posts) == 2  # 分 2 批


def test_api_embedding_token_and_base():
    e = ApiEmbedding(base="http://node:9000", api_key="s3cret")
    e.encode(["x"])
    url, body, headers = _FakeClient.instances[-1].posts[-1]
    assert url == "http://node:9000/embed"
    assert headers["X-Inference-Token"] == "s3cret"
    assert body["texts"] == ["x"]


def test_api_reranker_sorts_by_relevance():
    rk = ApiReranker(base="http://node:9000", api_key="tok")
    out = rk.rerank("q", [{"chunk_id": "a", "content": "x"}, {"chunk_id": "b", "content": "y"}])
    assert [c["chunk_id"] for c in out] == ["a", "b"]   # index0 (1.0) 排前
    assert out[0]["rank_score"] == 1.0
    assert out[1]["rank_score"] == 0.9


def test_api_reranker_falls_back_when_node_down(monkeypatch):
    class _Bad(_FakeClient):
        def post(self, url, json=None, headers=None):
            raise httpx.ConnectError("node down")

    monkeypatch.setattr(httpx, "Client", _Bad)
    rk = ApiReranker(base="http://node:9000")
    out = rk.rerank("q", [{"chunk_id": "a", "content": "x"}, {"chunk_id": "b", "content": "y"}])
    assert [c["chunk_id"] for c in out] == ["a", "b"]   # 节点不可达 -> 保持 RRF 原顺序,不抛异常
