"""托管嵌入/重排（票 38 / #47）：无 GPU 也能跑真实检索（SiliconFlow，bge-m3）。

两条要命的细节在这条链路上：
- **OpenAI 兼容接口不保证 `data[]` 与 `input[]` 同序** —— 直接按顺序取会把向量和文本错配。
  这种 bug 不报错，只是检索悄悄变差，所以必须有专门的用例钉住。
- **拿不到就说拿不到**：`data` 缺失/为空时抛错，不能静默返回空向量表
  （那会让整条检索退化成「什么都检索不到」，看上去却像正常跑完）。
"""
from __future__ import annotations

import httpx
import pytest

from app.core.embedding import SiliconFlowEmbedding, get_embedding
from app.core.reranker import SiliconFlowReranker, get_reranker


def _stub_httpx(monkeypatch, *, status=200, body=None, raises=False, capture=None):
    """替掉 httpx.Client；capture 收集每次请求的 url/json/headers 供断言。"""

    class Resp:
        def __init__(self, payload=None):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload if self._payload is not None else {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            if capture is not None:
                capture.append({"url": url, "json": json, "headers": headers})
            if raises:
                raise httpx.ConnectError("连不上")
            payload = body(json) if callable(body) else body      # 分批用例要按请求长度回
            return Resp(payload)

    monkeypatch.setattr(httpx, "Client", Client)


def _embed_body(pairs):
    """pairs = [(index, embedding)]，故意不保证顺序。"""
    return {"data": [{"index": i, "embedding": v} for i, v in pairs]}


def _settings(monkeypatch, **kw):
    from app.config import get_settings

    s = get_settings()
    for k, v in kw.items():
        monkeypatch.setattr(s, k, v)
    return s


# ---------- 嵌入 ----------

def test_the_default_base_is_siliconflow(monkeypatch):
    """不填 EMBEDDING_API_BASE 也能用 —— 默认指向 SiliconFlow，少一步配置。"""
    _settings(monkeypatch, embedding_api_base=None)
    assert SiliconFlowEmbedding().base == "https://api.siliconflow.cn/v1"


def test_an_explicit_base_url_wins(monkeypatch):
    """要接别家 OpenAI 兼容的托管服务，改一个环境变量就行。"""
    _settings(monkeypatch, embedding_api_base="https://api.example.com/v1/")
    assert SiliconFlowEmbedding().base == "https://api.example.com/v1"   # 末尾斜杠要削掉


def test_the_embedding_request_uses_the_openai_shape(monkeypatch):
    cap = []
    _stub_httpx(monkeypatch, body=_embed_body([(0, [0.1]), (1, [0.2])]), capture=cap)
    _settings(monkeypatch, embedding_api_base=None, embedding_api_key="sk-x",
              embedding_model="BAAI/bge-m3", embedding_dim=1)   # 夹具用 1 维向量，维度要对上

    SiliconFlowEmbedding().encode(["甲", "乙"])

    assert len(cap) == 1
    req = cap[0]
    assert req["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert req["headers"]["Authorization"] == "Bearer sk-x"        # Bearer，不是 X-Inference-Token
    assert req["json"]["model"] == "BAAI/bge-m3"                   # 模型必须随请求走
    assert req["json"]["input"] == ["甲", "乙"]


def test_out_of_order_embeddings_are_put_back_by_index(monkeypatch):
    """**最关键的一条**：接口不保证 data[] 与 input[] 同序，错配不会报错、只会悄悄变差。"""
    _stub_httpx(monkeypatch, body=_embed_body([(2, [3.0]), (0, [1.0]), (1, [2.0])]))
    _settings(monkeypatch, embedding_api_base=None, embedding_dim=1)

    got = SiliconFlowEmbedding().encode(["甲", "乙", "丙"])

    assert got == [[1.0], [2.0], [3.0]]      # 按 index 还原，不是按返回顺序


def test_embeddings_are_batched(monkeypatch):
    cap = []

    def echo(req):
        return _embed_body([(i, [float(i)]) for i in range(len(req["input"]))])

    _stub_httpx(monkeypatch, body=echo, capture=cap)
    _settings(monkeypatch, embedding_api_base=None, embedding_batch_size=2, embedding_dim=1)

    SiliconFlowEmbedding().encode(["a", "b", "c", "d", "e"])

    assert len(cap) == 3                     # 2 + 2 + 1
    assert [len(c["json"]["input"]) for c in cap] == [2, 2, 1]


def test_no_texts_means_no_request(monkeypatch):
    cap = []
    _stub_httpx(monkeypatch, capture=cap)
    _settings(monkeypatch, embedding_api_base=None)

    assert SiliconFlowEmbedding().encode([]) == []
    assert cap == []


def test_a_response_without_data_is_an_error(monkeypatch):
    """静默返回空 = 整条检索退化成「什么都检索不到」，看上去却像正常跑完。"""
    _stub_httpx(monkeypatch, body={"model": "x"})
    _settings(monkeypatch, embedding_api_base=None)

    with pytest.raises(RuntimeError):
        SiliconFlowEmbedding().encode(["甲"])


def test_an_empty_data_list_is_also_an_error(monkeypatch):
    _stub_httpx(monkeypatch, body={"data": []})
    _settings(monkeypatch, embedding_api_base=None)

    with pytest.raises(RuntimeError):
        SiliconFlowEmbedding().encode(["甲"])


def test_a_failed_embedding_call_raises(monkeypatch):
    """嵌入失败**不能**降级成空向量 —— 那等于假装检索过了。"""
    _stub_httpx(monkeypatch, raises=True)
    _settings(monkeypatch, embedding_api_base=None)

    with pytest.raises(Exception):
        SiliconFlowEmbedding().encode(["甲"])


# ---------- 重排 ----------

def test_the_rerank_request_uses_the_cohere_flat_shape(monkeypatch):
    cap = []
    _stub_httpx(monkeypatch, body={"results": [{"index": 0, "relevance_score": 0.9}]}, capture=cap)
    _settings(monkeypatch, rerank_api_base=None, rerank_api_key="sk-y",
              reranker_model="BAAI/bge-reranker-v2-m3")

    SiliconFlowReranker().rerank("营收", [{"content": "甲"}, {"content": "乙"}])

    req = cap[0]
    assert req["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert req["headers"]["Authorization"] == "Bearer sk-y"
    assert req["json"] == {"model": "BAAI/bge-reranker-v2-m3", "query": "营收",
                           "documents": ["甲", "乙"], "top_n": 2}


def test_rerank_reorders_by_relevance_score(monkeypatch):
    _stub_httpx(monkeypatch, body={"results": [{"index": 0, "relevance_score": 0.1},
                                                {"index": 1, "relevance_score": 0.9}]})
    _settings(monkeypatch, rerank_api_base=None)

    out = SiliconFlowReranker().rerank("q", [{"content": "甲"}, {"content": "乙"}])

    assert [c["content"] for c in out] == ["乙", "甲"]
    assert out[0]["rank_score"] == 0.9


def test_documents_the_provider_did_not_score_get_zero(monkeypatch):
    _stub_httpx(monkeypatch, body={"results": [{"index": 1, "relevance_score": 0.8}]})
    _settings(monkeypatch, rerank_api_base=None)

    out = SiliconFlowReranker().rerank("q", [{"content": "甲"}, {"content": "乙"}])

    assert out[0]["rank_score"] == 0.8
    assert out[1]["rank_score"] == 0.0       # 没被评分的记 0，不猜


def test_a_failed_rerank_falls_back_to_the_rrf_order(monkeypatch):
    """节点不可达不能打断回答：按 RRF 原顺序返回（与既有 api provider 一致）。"""
    _stub_httpx(monkeypatch, raises=True)
    _settings(monkeypatch, rerank_api_base=None)
    candidates = [{"content": "甲"}, {"content": "乙"}]

    out = SiliconFlowReranker().rerank("q", candidates)

    assert [c["content"] for c in out] == ["甲", "乙"]


def test_no_candidates_means_no_request(monkeypatch):
    cap = []
    _stub_httpx(monkeypatch, capture=cap)
    _settings(monkeypatch, rerank_api_base=None)

    assert SiliconFlowReranker().rerank("q", []) == []
    assert cap == []


# ---------- 工厂 ----------

def test_the_factories_pick_the_hosted_providers(monkeypatch):
    _settings(monkeypatch, embedding_api_base=None, rerank_api_base=None)

    assert isinstance(get_embedding("siliconflow"), SiliconFlowEmbedding)
    assert isinstance(get_reranker("siliconflow"), SiliconFlowReranker)


def test_the_existing_providers_are_untouched(monkeypatch):
    from app.core.embedding import FakeEmbedding
    from app.core.reranker import FakeReranker

    assert isinstance(get_embedding("fake"), FakeEmbedding)
    assert isinstance(get_reranker("fake"), FakeReranker)


def test_a_missing_embedding_in_a_data_item_is_an_error(monkeypatch):
    """**两轴都抓到的硬违规**：空向量放行后 cosine 恒为 0，那段内容就永久检索不到了，
    而且全程不报错 —— 比直接失败更难查。"""
    _stub_httpx(monkeypatch, body={"data": [{"index": 0}]})       # 有条目、没 embedding
    _settings(monkeypatch, embedding_api_base=None)

    with pytest.raises(RuntimeError):
        SiliconFlowEmbedding().encode(["甲"])


def test_an_explicitly_empty_embedding_is_also_an_error(monkeypatch):
    _stub_httpx(monkeypatch, body={"data": [{"index": 0, "embedding": []}]})
    _settings(monkeypatch, embedding_api_base=None)

    with pytest.raises(RuntimeError):
        SiliconFlowEmbedding().encode(["甲"])


def test_a_dimension_mismatch_is_an_error_not_a_silent_wrong_score(monkeypatch):
    """维度对不上时 cosine 会在短的一侧截断 —— 算出错的相似度却不报错。"""
    _stub_httpx(monkeypatch, body=_embed_body([(0, [0.1, 0.2, 0.3])]))   # 3 维
    _settings(monkeypatch, embedding_api_base=None, embedding_dim=1024)

    with pytest.raises(RuntimeError) as e:
        SiliconFlowEmbedding().encode(["甲"])
    assert "EMBEDDING_DIM" in str(e.value)          # 报错要告诉人怎么改
