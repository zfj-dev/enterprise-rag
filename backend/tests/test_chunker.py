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


def test_page_unique_ids():
    """多页按页分块时，不同页的 chunk id 不允许重复（历史 bug：页间 id 碰撞）。"""
    ch = ParentChildChunker(parent_size=60, child_size=20, overlap=5)
    a = ch.chunk("第一页内容。测试分块。", doc_id="d1", page_num=1)
    b = ch.chunk("第二页内容。测试分块。", doc_id="d1", page_num=2)
    ids = [c["id"] for c in a + b]
    assert len(ids) == len(set(ids)), "不同页的 chunk id 不应重复"


def test_table_kept_whole():
    ch = ParentChildChunker(parent_size=200, child_size=50, overlap=5)
    table = "表3.1 实验结果对比\n| 模型 | 精确率 | 召回率 |\n| --- | --- | --- |\n" + "\n".join(f"| YOLOv8-{i} | {90+i}% | {80+i}% |" for i in range(1, 6))
    text = "本系统采用YOLOv8算法。\n" + table + "\n综上，识别效果良好。"
    chunks = ch.chunk(text, doc_id="d1", page_num=1)
    tbl_child = [c for c in chunks if c["chunk_type"] == "child" and c["content"].startswith("表3.1")]
    assert tbl_child, "应有一个包含表格的 child"
    assert "YOLOv8-5" in tbl_child[0]["content"], "表格应保持完整不被拆散"
    assert tbl_child[0]["content"] == tbl_child[0]["parent_content"]


def test_table_caption_merged_with_blank_line():
    """docling 输出：表格标题与表格之间隔空行，标题应并入表格单元（否则"表3.1"查不到正文）。"""
    ch = ParentChildChunker(parent_size=512, child_size=128, overlap=20)
    text = ("前文内容。\n"
            "表 3 . 1 实验环境配置\n"
            "\n"
            "| 实验环境 | 名称 |\n"
            "| --- | --- |\n"
            "| 代码编译器 | PyCharm |\n"
            "| GPU | RTX 3060 |\n")
    chunks = ch.chunk(text, doc_id="d1", page_num=1)
    tbl_child = [c for c in chunks if c["chunk_type"] == "child" and "RTX 3060" in c["content"]]
    assert tbl_child, "表格 chunk 应存在"
    assert "表 3 . 1" in tbl_child[0]["content"], "标题应并入表格 chunk"
    assert "PyCharm" in tbl_child[0]["content"], "表格正文应完整"
