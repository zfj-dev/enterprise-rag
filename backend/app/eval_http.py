"""评测脚本共用的 HTTP 小工具 —— 上传并等入库这类活只该写一遍。"""
from __future__ import annotations

import os
import time


def upload_and_wait(client, kb_id: str, headers: dict, path: str, wait_seconds: int = 240) -> dict:
    """上传被评文档并等它入库，返回 {"status", "chunk_count", "page_count"}。

    超时未出结果也照样返回最后一瞥的状态 —— 调用方自己决定怎么报，别在这里吞掉。
    """
    with open(path, "rb") as f:
        up = client.post("/api/v1/documents?kb_id=%s" % kb_id, headers=headers,
                         files={"file": (os.path.basename(path), f, "application/pdf")}).json()
    doc = {}
    for _ in range(wait_seconds):
        doc = client.get("/api/v1/documents/%s" % up["id"], headers=headers).json()
        if doc.get("status") in ("indexed", "failed"):
            break
        time.sleep(1)
    return doc
