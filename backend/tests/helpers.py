"""测试共用小工具。"""
from __future__ import annotations

import json


def sse_events(text: str) -> list[dict]:
    """把 SSE 响应体解析成事件列表（无法解析的 data: 行直接跳过）。"""
    out: list[dict] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except Exception:
                continue
    return out
