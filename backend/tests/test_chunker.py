from app.core.chunker import ParentChildChunker


def test_parent_child_split():
    ch = ParentChildChunker(parent_size=60, child_size=20, overlap=5)
    text = "这是第一句。第二句有点长，用来测试分块逻辑。第三句结束。"
    chunks = ch.chunk(text, doc_id="d1", page_num=3)
    children = [c for c in chunks if c["chunk_type"] == "child"]
    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    assert children and parents
    assert all(c["page_num"] == 3 for c in chunks)
    parent_ids = {p["id"] for p in parents}
    assert all(c["parent_id"] in parent_ids for c in children)
    assert all(c["parent_content"] for c in children)


def test_contextual_summary_prefix():
    ch = ParentChildChunker(parent_size=60, child_size=20, overlap=5, contextual_summary=True)
    chunks = ch.chunk("测试文本内容。", doc_id="d1", doc_summary="关于测试的文档")
    child = [c for c in chunks if c["chunk_type"] == "child"][0]
    assert "关于测试的文档" in child["content"]
