"""测试共用小工具。"""
from __future__ import annotations

import json
import time


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


def wait_until(pred, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """轮询等待条件成立 —— 后台线程落库这类异步副作用的测试用。"""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False
