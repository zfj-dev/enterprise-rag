"""文档解析路由：按文件类型分派到不同解析器，统一输出结构化文本。

原生文字走 PyMuPDF / python-docx / openpyxl（快）；复杂表格/扫描件在真实部署可升级 Docling/PP-Structure
（见架构文档 §5，此处保留路由扩展点）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    text: str
    pages: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    if ext == "pdf":
        return "pdf"
    if ext in ("docx", "doc"):
        return "docx"
    if ext in ("xlsx", "xls", "csv"):
        return "xlsx"
    if ext in ("md", "markdown"):
        return "md"
    if ext == "txt":
        return "txt"
    if ext in ("png", "jpg", "jpeg", "gif", "bmp"):
        return "image"
    return "unknown"


def _parse_pdf(path: str) -> str:
    import fitz  # PyMuPDF，惰性

    text, page_count = [], 0
    with fitz.open(path) as doc:
        page_count = doc.page_count
        for page in doc:
            text.append(page.get_text("text"))
    return "\n\n".join(text), page_count


def _parse_docx(path: str) -> str:
    import docx  # python-docx，惰性

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(cells))
    return "\n".join(parts), len(d.paragraphs)


def _parse_xlsx(path: str) -> str:
    import openpyxl  # 惰性

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"# 工作表: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            vals = [str(c) for c in row if c is not None]
            if vals:
                parts.append(" | ".join(vals))
    return "\n".join(parts), wb.sheetnames  # type: ignore


def _parse_text(path: str) -> str:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    return raw, raw.count("\n")


class ParserRouter:
    """按 detect_kind 分发；未知/图片返回空文本并标记（真实部署接 OCR/多模态）。"""

    def parse(self, path: str, filename: str | None = None) -> ParsedDocument:
        name = filename or os.path.basename(path)
        kind = detect_kind(name)
        try:
            if kind == "pdf":
                text, meta = _parse_pdf(path)
                return ParsedDocument(text=text, pages=meta, metadata={"kind": "pdf"})
            if kind == "docx":
                text, meta = _parse_docx(path)
                return ParsedDocument(text=text, pages=meta if isinstance(meta, int) else 0,
                                      metadata={"kind": "docx"})
            if kind == "xlsx":
                text, meta = _parse_xlsx(path)
                return ParsedDocument(text=text, pages=0, metadata={"kind": "xlsx"})
            if kind in ("md", "txt"):
                text, pages = _parse_text(path)
                return ParsedDocument(text=text, pages=pages, metadata={"kind": kind})
            if kind == "image":
                return ParsedDocument(text="", pages=1, metadata={"kind": "image", "note": "需 OCR/多模态"})
            return ParsedDocument(text="", pages=0, metadata={"kind": "unknown", "error": "不支持的文件类型"})
        except ImportError as e:
            return ParsedDocument(text="", pages=0,
                                  metadata={"kind": kind, "error": f"缺少解析依赖: {e.name}"})
