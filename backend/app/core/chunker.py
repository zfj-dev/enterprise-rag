"""Parent-Child 分块：子块(检索) + 父块(生成)，支持前置来源上下文摘要(contextual retrieval)。

本 MVP 用字符宽度做粗略分块（无外挂 tokenizer 也可运行/测试）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

_SENT_END = re.compile(r"(?<=[。！？!?\n])")


def _sentences(text: str) -> list[str]:
    parts = _SENT_END.split(text)
    return [p for p in parts if p.strip()]


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
        while start < len(text):
            out.append(text[start:start + window])
            if start + window >= len(text):
                break
            start += step
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

    def chunk(self, text: str, doc_id: str, page_num: int = 0, doc_summary: str | None = None) -> list[dict]:
        prefix = f"[文档概要] {doc_summary}\n" if (self.contextual and doc_summary) else ""
        parents = self._semantic_parents(text)
        chunks: list[dict] = []

        for p_idx, parent in enumerate(parents):
            parent_id = f"{doc_id}_p{p_idx}"
            children = self._split_window(parent, self.child_size, self.overlap)
            for c_idx, child in enumerate(children):
                child_content = (prefix + child) if prefix else child
                chunks.append({
                    "id": f"{parent_id}_c{c_idx}", "parent_id": parent_id,
                    "content": child_content, "parent_content": parent,
                    "chunk_type": "child", "page_num": page_num, "doc_id": doc_id,
                })
            # 父块也入库（生成时加载完整上下文）
            chunks.append({
                "id": parent_id, "parent_id": None,
                "content": (prefix + parent) if prefix else parent, "parent_content": parent,
                "chunk_type": "parent", "page_num": page_num, "doc_id": doc_id,
            })
        return chunks
