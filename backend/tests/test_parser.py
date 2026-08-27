from app.core.parser import ParserRouter, detect_kind


def test_detect_kind():
    assert detect_kind("a.pdf") == "pdf"
    assert detect_kind("b.docx") == "docx"
    assert detect_kind("c.md") == "md"
    assert detect_kind("d.txt") == "txt"
    assert detect_kind("e.xlsx") == "xlsx"
    assert detect_kind("f.png") == "image"
    assert detect_kind("g.xyz") == "unknown"


def test_parse_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("你好世界\n第二行", encoding="utf-8")
    pd = ParserRouter().parse(str(p), "note.txt")
    assert "你好世界" in pd.text


def test_parse_md(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# 标题\n正文内容", encoding="utf-8")
    pd = ParserRouter().parse(str(p), "doc.md")
    assert "正文内容" in pd.text


def test_parse_unknown(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_bytes(b"data")
    pd = ParserRouter().parse(str(p), "x.xyz")
    assert pd.text == ""
    assert pd.metadata.get("error")
