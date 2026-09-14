"""评测脚本共用的 HTTP 小工具 —— 上传并等入库这类活只该写一遍。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass


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

@dataclass
class OnlineSession:
    """打**服务端**评测要的那点前置：登录 + 一个临时知识库 + 一份已入库的文档。

    生成层与延迟段都要这一套 —— 各做一遍就是**把同一份 PDF 解析、嵌入两次**（#55）。
    谁开的谁关：`close()` 会删掉那个临时库。
    """

    client: object            # httpx.Client
    headers: dict
    kb_id: str
    upload: dict

    @classmethod
    def open(cls, base: str, doc_path: str, name: str = "__eval__") -> "OnlineSession":
        import httpx

        c = httpx.Client(base_url=base, timeout=300)
        r = c.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        r.raise_for_status()
        headers = {"Authorization": "Bearer %s" % r.json().get("access_token")}
        kb_id = c.post("/api/v1/knowledge", json={"name": name, "description": ""},
                       headers=headers).json()["id"]
        upload = upload_and_wait(c, kb_id, headers, doc_path)
        return cls(client=c, headers=headers, kb_id=kb_id, upload=upload)

    def close(self) -> None:
        """删掉临时库并关连接 —— 失败也不抛（关不干净不该盖住评测结果）。"""
        try:
            self.client.delete("/api/v1/knowledge/%s" % self.kb_id, headers=self.headers)
        except Exception:   # noqa: BLE001
            pass
        finally:
            self.client.close()
