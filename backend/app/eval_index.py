"""把切好的块灌进内存索引 —— 评测脚本共用一份，别各写一遍。

检索评测（evaluate_retrieval）与 RGB 评测（evaluate_rgb）都要做
「切块 → 嵌入 → 进向量库 + BM25」这件事，嵌入式索引的写法只该有一处。
"""
from __future__ import annotations


def index_chunks(rt, chunks: list) -> int:
    """child 块 → 嵌入 → 进向量库 + BM25，返回块数。

    chunks 每条：{"id", "content", "metadata"}。metadata 里要带
    kb_id / owner_id / doc_name / page_num / content —— 检索时的权限过滤与引用渲染都要用。
    """
    if not chunks:
        return 0
    from app.core.vector_store import VectorItem

    vectors = rt.embedding.encode([c["content"] for c in chunks])
    rt.vector_store.add([VectorItem(id=c["id"], vector=v, metadata=dict(c["metadata"]))
                         for c, v in zip(chunks, vectors)])
    rt.bm25.add([{"id": c["id"], "content": c["content"], "metadata": dict(c["metadata"])}
                 for c in chunks])
    return len(chunks)
