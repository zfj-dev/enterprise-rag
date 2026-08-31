"""文档解析路由：按文件类型分派，统一输出文本 + 分页文本（用于按页分块/标注页码）。"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings


@dataclass
class ParsedDocument:
    text: str
    page_count: int = 0
    pages: list[str] = field(default_factory=list)  # PDF 逐页；其他格式单元素
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


_docling_converter = None
_docling_lock = threading.Lock()


def _build_docling_converter():
    """构建 DocumentConverter（惰性 import；只构建一次，模型常驻内存）。

    - 后端：PyPdfium（绕开 docling_parse 的 additional.dat 缺失问题）。
    - OCR：关闭（文本型 PDF 不需要，OCR 极慢）。
    - device：优先 cuda（可用时），否则 cpu；可被 DOCLING_DEVICE 环境变量覆盖。
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption  # 惰性
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    dev = os.environ.get("DOCLING_DEVICE", "")
    if not dev:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_opts = PdfPipelineOptions(
        do_ocr=False,  # 文本型 PDF 不需要 OCR（大提速）
        do_formula_enrichment=get_settings().docling_formula_enrichment,
        images_scale=get_settings().docling_images_scale,
        table_structure_options=TableStructureOptions(mode=get_settings().docling_table_mode),
        accelerator_options=AcceleratorOptions(device=dev, num_threads=8),
    )
    print(f"[doc] docling converter: device={dev} ocr=False")
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipe_opts,
            ),
        }
    )


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is None:
        with _docling_lock:
            if _docling_converter is None:
                _docling_converter = _build_docling_converter()
    return _docling_converter


def _parse_pdf_docling(path: str):
    """Docling 解析 PDF（表格/版面/层级更强）。惰性 import，失败由调用方回退 PyMuPDF。

    用 PyPdfium 后端而非默认 docling_parse：docling_parse 的 Windows wheel 常缺
    pdf_resources/glyphs/standard/additional.dat（字体表）导致 RuntimeError；
    pdfium 不依赖该文件，版面/表格仍走 StandardPdfPipeline（需 onnxruntime）。
    复用模块级 converter：模型只加载一次，后续解析只需推理。

    按页导出 markdown（export_to_markdown(page_no=...)），使 chunk 带真实页码，
    引用溯源不再全显示"第1页"。
    """
    import os

    converter = _get_docling_converter()
    with _docling_lock:
        res = converter.convert(os.path.abspath(path))  # 绝对路径更稳

    pages_map = getattr(res.document, "pages", None) or {}
    page_objs = list(pages_map.values()) if isinstance(pages_map, dict) else list(pages_map)
    page_count = len(page_objs) or 1
    pages: list[str] = []
    for p in page_objs:
        pno = getattr(p, "page_no", None)
        try:
            md_page = res.document.export_to_markdown(page_no=pno) if pno is not None else ""
        except Exception:  # noqa
            md_page = ""
        pages.append(md_page or "")  # 空页保留占位，保证 page_num 与真实页一致
    if not any(pg.strip() for pg in pages):  # 极端兜底：整篇单段
        pages = [res.document.export_to_markdown()]
    text = "\n\n".join(pages)
    return ParsedDocument(text=text, page_count=page_count, pages=pages,
                          metadata={"kind": "pdf", "parser": "docling"})


def _parse_pdf(path: str):
    import fitz  # PyMuPDF

    page_texts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_texts.append(page.get_text("text"))
    return "\n\n".join(page_texts), len(page_texts), page_texts


def _parse_docx(path: str):
    import docx

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return text, len(d.paragraphs)


def _parse_xlsx(path: str):
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"# 工作表: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            vals = [str(c) for c in row if c is not None]
            if vals:
                parts.append(" | ".join(vals))
    return "\n".join(parts), len(wb.sheetnames)


def _parse_text(path: str):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    return raw, raw.count("\n")


class ParserRouter:
    def parse(self, path: str, filename: str | None = None) -> ParsedDocument:
        name = filename or os.path.basename(path)
        kind = detect_kind(name)
        try:
            if kind == "pdf":
                if get_settings().parser_use_docling:
                    try:
                        return _parse_pdf_docling(path)
                    except Exception as e:  # noqa
                        import traceback
                        try:
                            import docling
                            _v = getattr(docling, "__version__", "?")
                        except Exception:
                            _v = "?"
                        print(f"[doc] docling({_v}) 失败，回退 PyMuPDF: {e}\n{traceback.format_exc()}")
                text, count, pages = _parse_pdf(path)
                return ParsedDocument(text=text, page_count=count, pages=pages, metadata={"kind": "pdf"})
            if kind == "docx":
                text, count = _parse_docx(path)
                return ParsedDocument(text=text, page_count=count, pages=[text], metadata={"kind": "docx"})
            if kind == "xlsx":
                text, count = _parse_xlsx(path)
                return ParsedDocument(text=text, page_count=count, pages=[text], metadata={"kind": "xlsx"})
            if kind in ("md", "txt"):
                text, count = _parse_text(path)
                return ParsedDocument(text=text, page_count=count, pages=[text], metadata={"kind": kind})
            if kind == "image":
                return ParsedDocument(text="", page_count=1, pages=[""], metadata={"kind": "image", "note": "需 OCR/多模态"})
            return ParsedDocument(text="", page_count=0, pages=[], metadata={"kind": "unknown", "error": "不支持的文件类型"})
        except ImportError as e:
            return ParsedDocument(text="", page_count=0, pages=[],
                                  metadata={"kind": kind, "error": f"缺少解析依赖: {e.name}"})
