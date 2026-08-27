"""端到端 API 集成测试：注册→登录→建库→上传文档→入库→流式问答→反馈→调试。"""
from __future__ import annotations


def _token(client, username: str) -> str:
    r = client.post("/api/v1/auth/register", json={"username": username, "password": "pw123456"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_health(client):
    r = client.get("/api/v1/auth/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_flow(client):
    tok = _token(client, "alice")
    assert tok
    ok = client.post("/api/v1/auth/login", json={"username": "alice", "password": "pw123456"})
    assert ok.status_code == 200
    bad = client.post("/api/v1/auth/login", json={"username": "alice", "password": "bad"})
    assert bad.status_code == 401


def test_kb_upload_chat_feedback_debug(client):
    tok = _token(client, "bob")
    H = {"Authorization": f"Bearer {tok}"}

    # 建知识库
    kr = client.post("/api/v1/knowledge", json={"name": "产品手册", "description": "测试"}, headers=H)
    assert kr.status_code == 200, kr.text
    kb_id = kr.json()["id"]

    # 上传文档（多模态文本）→ 入库
    files = {"file": ("manual.txt", "比亚迪安全手册，关于电池和充电的规范说明。", "text/plain")}
    up = client.post(f"/api/v1/documents?kb_id={kb_id}", headers=H, files=files)
    assert up.status_code == 200, up.text
    doc = up.json()
    assert doc["status"] == "indexed"
    assert doc["chunk_count"] > 0

    # 流式问答
    cr = client.post("/api/v1/chat/stream", headers=H,
                     json={"kb_id": kb_id, "question": "比亚迪 电池 规范", "stream": True})
    assert cr.status_code == 200, cr.text
    body = cr.text
    assert "data:" in body
    assert "sources" in body

    # 反馈
    # 先解析出 message_id（取最后一个 done 事件）
    import json
    msg_id = None
    for line in cr.text.splitlines():
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
            except Exception:
                continue
            if data.get("type") == "done":
                msg_id = data.get("message_id")
    assert msg_id
    fr = client.post("/api/v1/feedback", headers=H, json={"message_id": msg_id, "rating": 1})
    assert fr.status_code == 200, fr.text

    # 调试面板（检索中间结果）
    dr = client.get(f"/api/v1/debug/query?kb_id={kb_id}&question=比亚迪电池", headers=H)
    assert dr.status_code == 200, dr.text
    assert "retrieval_top" in dr.json()
