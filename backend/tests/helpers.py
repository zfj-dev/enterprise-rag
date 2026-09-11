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


def register_and_kb(client, name: str):
    """注册一个用户并给他建一个同名知识库，返回 (headers, user_id, kb_id)。"""
    from app.db.session import SessionLocal
    from app.models.entities import User

    tok = client.post("/api/v1/auth/register",
                      json={"username": name, "password": "pw123456"}).json()["access_token"]
    H = {"Authorization": "Bearer {}".format(tok)}
    kb_id = client.post("/api/v1/knowledge", json={"name": "{}-kb".format(name), "description": ""},
                        headers=H).json()["id"]
    db = SessionLocal()
    try:
        uid = db.query(User).filter(User.username == name).first().id
    finally:
        db.close()
    return H, uid, kb_id


class StubEmbedding:
    """确定性的嵌入替身：含关键词给 [1,0]，否则零向量 —— 相似度只可能是 1 或 0。

    记忆召回按相似度排序，用它在测试里就没有阈值歧义。
    """

    KEY = "电池"

    def encode(self, texts):
        return [[1.0, 0.0] if self.KEY in t else [0.0, 0.0] for t in texts]


def wait_until(pred, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """轮询等待条件成立 —— 后台线程落库这类异步副作用的测试用。"""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False
