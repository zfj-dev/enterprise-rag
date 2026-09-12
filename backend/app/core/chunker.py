"""Parent-Child 分块：子块(检索) + 父块(生成)，支持前置来源上下文摘要与表格感知。

chunk id 含 doc_id + page_num + 块序号，保证"多页文档按页分块"时全局唯一。
表格块（连续以 | 开头的 Markdown 表格行）保持完整不拆散，便于检索与生成。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

_SENT_END = re.compile(r"(?<=[。！？!?\n])")

_FORMULA_SPAN_RE = re.compile(r"\$\$[\s\S]+?\$\$|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)")


def _formula_spans(text: str) -> list[tuple[int, int]]:
    """返回 LaTeX 公式块（$$..$$ 或 $..$）在文本中的 [start,end) 区间（含定界符）。

    用于分块时把公式当作原子块：不让 `_split_window` 从公式中间拦腰截断（长公式 > child_size 之前会被切坏）。
    """
    if not text:
        return []
    return [(m.start(), m.end()) for m in _FORMULA_SPAN_RE.finditer(text)]


def _sentences(text: str) -> list[str]:
    parts = _SENT_END.split(text)
    return [p for p in parts if p.strip()]


def _looks_like_caption(line: str) -> bool:
    """判断一行是否为表格/图标题（如 `表3.1 xxx`、`表 3 . 1 实验环境配置`、`Table 3.1`）。

    去空格后以 表/图/Table/Fig + 数字 开头即视为标题，用于把标题并入表格单元。
    """
    compact = re.sub(r"\s+", "", line or "")
    return bool(re.match(r"^(表|图|Table|Fig)\s*[.\-]?\s*\d", compact, re.IGNORECASE))


@dataclass
class ChunkUnit:
    content: str
    parent_content: str
    chunk_type: str
    parent_id: str | None
    page_num: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ParentChildChunker:
    def __init__(self, parent_size: int | None = None, child_size: int | None = None,
                 overlap: int | None = None, contextual_summary: bool | None = None):
        s = get_settings()
        self.parent_size = parent_size or s.chunk_parent_size
        self.child_size = child_size or s.chunk_child_size
        self.overlap = overlap if overlap is not None else s.chunk_overlap
        self.contextual = contextual_summary if contextual_summary is not None else s.contextual_summary

    def _split_window(self, text: str, window: int, overlap: int) -> list[str]:
        out, start = [], 0
        step = max(window - overlap, 1)
        n = len(text)
        spans = _formula_spans(text)
        while start < n:
            # start 若落在公式内部 -> 跳到公式末尾（该公式已被前一窗口完整捕获，避免重复）
            for s, e in spans:
                if s < start < e:
                    start = e
                    break
            if start >= n:
                break
            end = min(start + window, n)
            # end 若切断公式 -> 扩到公式结束，保证公式完整不残缺
            for s, e in spans:
                if s < end < e:
                    end = e
                    break
            if end <= start:  # 兜底防死循环
                end = min(start + window, n)
            out.append(text[start:end])
            if end >= n:
                break
            nxt = max(end - overlap, start + step)
            if nxt <= start:  # 兜底防死循环
                nxt = start + step
            start = nxt
        return out

    def _semantic_parents(self, text: str) -> list[str]:
        parents, cur = [], ""
        for sent in _sentences(text):
            if cur and len(cur) + len(sent) > self.parent_size:
                parents.append(cur)
                cur = sent
            else:
                cur += sent
        if cur:
            parents.append(cur)
        return parents

    def _table_aware_parents(self, text: str) -> list[tuple[str, bool]]:
        """按行拆分，把连续以 '|' 开头的表格行聚成一个不拆分的单元。返回 [(content, is_table)]。

        若表格前（隔空行）紧邻表格标题（如 `表 3 . 1 实验环境配置`），把标题并入表格单元，
        保证"表3.1"这类引用能检索到表格正文。
        """
        lines = text.split("\n")
        units: list[tuple[str, bool]] = []
        cur: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("|"):
                caption_lines: list[str] = []
                if cur:
                    # 跳过表格前的空行
                    while cur and not cur[-1].strip():
                        cur.pop()
                    # 把紧邻表格前面的标题行并入表格单元
                    while cur and _looks_like_caption(cur[-1]):
                        caption_lines.insert(0, cur.pop())
                    for p in self._semantic_parents("\n".join(cur)):
                        if p.strip():
                            units.append((p, False))
                    cur = []
                tbl = caption_lines + [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tbl.append(lines[i])
                    i += 1
                tbl_text = "\n".join(tbl).strip()
                if tbl_text:
                    units.append((tbl_text, True))
            else:
                cur.append(line)
                i += 1
        if cur:
            for p in self._semantic_parents("\n".join(cur)):
                if p.strip():
                    units.append((p, False))
        return units

    def chunk(self, text: str, doc_id: str, page_num: int = 0, doc_summary: str | None = None) -> list[dict]:
        prefix = f"[文档概要] {doc_summary}\n" if (self.contextual and doc_summary) else ""
        units = self._table_aware_parents(text)
        chunks: list[dict] = []

        for p_idx, (parent, is_table) in enumerate(units):
            parent_id = f"{doc_id}_p{page_num}_{p_idx}"
            children = [parent] if is_table else self._split_window(parent, self.child_size, self.overlap)
            for c_idx, child in enumerate(children):
                child_content = (prefix + child) if prefix else child
                chunks.append({
                    "id": f"{parent_id}_c{c_idx}", "parent_id": parent_id,
                    "content": child_content, "parent_content": parent,
                    "chunk_type": "child", "page_num": page_num, "doc_id": doc_id,
                })
            chunks.append({
                "id": parent_id, "parent_id": None,
                "content": (prefix + parent) if prefix else parent, "parent_content": parent,
                "chunk_type": "parent", "page_num": page_num, "doc_id": doc_id,
            })
        return chunks
