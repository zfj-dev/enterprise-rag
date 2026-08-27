from app.core.retriever import rrf_fuse, HybridRetriever
from app.core.embedding import FakeEmbedding
from app.core.vector_store import InMemoryVectorStore, VectorItem
from app.core.bm25 import InMemoryBm25


def test_rrf_fuse_dedup_order():
    v = [{"chunk_id": "a", "content": "x"}, {"chunk_id": "b", "content": "y"}]
    b = [{"chunk_id": "b", "content": "y"}, {"chunk_id": "c", "content": "z"}]
    merged = rrf_fuse(v, b, k=60, top_k=10)
    assert {m["chunk_id"] for m in merged} == {"a", "b", "c"}
    # b 在两边都出现 → 融合分应最高
    assert merged[0]["chunk_id"] == "b"


def test_hybrid_retriever_metadata_filter():
    emb = FakeEmbedding(dim=64)
    vs = InMemoryVectorStore()
    bm25 = InMemoryBm25()
    vs.add([
        VectorItem(id="c1", vector=emb.encode(["比亚迪安全手册 电池 充电"])[0],
                   metadata={"kb_id": "kb1", "owner_id": "u1", "doc_name": "doc1",
                             "content": "比亚迪安全手册 电池 充电", "page_num": 3}),
        VectorItem(id="c2", vector=emb.encode(["华为流程 报销"])[0],
                   metadata={"kb_id": "kb2", "owner_id": "u1", "doc_name": "doc2",
                             "content": "华为流程 报销", "page_num": 1}),
    ])
    bm25.add([
        {"id": "c1", "content": "比亚迪安全手册 电池 充电",
         "metadata": {"kb_id": "kb1", "owner_id": "u1", "content": "比亚迪安全手册 电池 充电"}},
        {"id": "c2", "content": "华为流程 报销",
         "metadata": {"kb_id": "kb2", "owner_id": "u1", "content": "华为流程 报销"}},
    ])
    rt = HybridRetriever(vs, bm25, emb, reranker=None)
    hits = rt.retrieve("比亚迪 电池", kb_id="kb1", owner_id="u1")
    assert hits
    assert all(h["metadata"]["kb_id"] == "kb1" for h in hits)
    assert all(h["metadata"]["owner_id"] == "u1" for h in hits)
