"""构造 chunk 的元数据（检索时用于权限过滤 + 引用溯源）。"""
from __future__ import annotations

from typing import Any


def make_meta(doc, content: str, page_num: int = 0) -> dict[str, Any]:
    return {
        "kb_id": doc.kb_id,
        "owner_id": doc.owner_id,
        "doc_id": doc.id,
        "doc_name": doc.filename,
        "page_num": page_num,
        "content": content,
    }
