from app.services.chat_service import (
    _enum_intent, _looks_like_table, _looks_like_figure,
    _looks_like_formula, _looks_like_code,
)


def test_enum_intent():
    # 枚举表格
    assert _enum_intent("给出所有表格内容") == "table"
    assert _enum_intent("列出全部表格") == "table"
    assert _enum_intent("有哪些表格") == "table"
    # 枚举图
    assert _enum_intent("有哪些图片") == "figure"
    assert _enum_intent("列出所有图") == "figure"
    # 枚举公式 / 代码
    assert _enum_intent("列出全部公式") == "formula"
    assert _enum_intent("给出所有代码") == "code"
    # 具体编号引用 → 不触发（走普通检索）
    assert _enum_intent("表3.1的内容是什么") is None
    assert _enum_intent("图 2 . 1 讲了什么") is None
    # 无关 / 非枚举
    assert _enum_intent("比亚迪2025年营业收入") is None
    assert _enum_intent("资产负债表怎么看") is None


def test_chunk_classifiers():
    assert _looks_like_table("表 3 . 1 实验环境配置\n| 实验环境 | 名称 |\n| GPU | RTX 3060 |")
    assert not _looks_like_table("本章节所有实验均在表3.1所示环境下验证")
    assert _looks_like_figure("<!-- image -->\n图 3 . 4 CBAM 模块")
    assert _looks_like_formula("<!-- formula-not-decoded -->")
    assert _looks_like_formula("$$E = mc^2$$")                            # docling 原始公式文本($$..$$)
    assert _looks_like_formula("x = \\begin{equation} y \\end{equation}")  # LaTeX 环境
    assert _looks_like_formula("inline \\(a+b\\)")                     # LaTeX 行内
    assert not _looks_like_formula("表 3 . 1 实验环境配置")                 # 不误判表格
    assert _looks_like_code("```python\nprint(1)\n```")
