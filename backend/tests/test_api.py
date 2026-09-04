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
    bad = client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong091"})
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
    assert doc["status"] in ("processing", "pending")   # 异步上传：先返回 processing
    doc = _wait_indexed(client, H, doc["id"])
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


def test_delete_doc_and_kb(client):
    tok = _token(client, "carol")
    H = {"Authorization": f"Bearer {tok}"}
    kb = client.post("/api/v1/knowledge", json={"name": "待删库", "description": ""}, headers=H).json()["id"]
    files = {"file": ("a.txt", "测试内容 abcdef", "text/plain")}
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H, files=files).json()
    doc_id = up["id"]
    up = _wait_indexed(client, H, doc_id)
    assert up["status"] == "indexed"

    dr = client.delete(f"/api/v1/documents/{doc_id}", headers=H)
    assert dr.status_code == 200
    lst = client.get(f"/api/v1/documents?kb_id={kb}", headers=H).json()
    assert all(d["id"] != doc_id for d in lst)

    kr = client.delete(f"/api/v1/knowledge/{kb}", headers=H)
    assert kr.status_code == 200
    kbs = client.get("/api/v1/knowledge", headers=H).json()
    assert all(k["id"] != kb for k in kbs)

import time


def _wait_indexed(client, H, doc_id, timeout=20):
    t0 = time.time()
    d = {}
    while time.time() - t0 < timeout:
        d = client.get(f"/api/v1/documents/{doc_id}", headers=H).json()
        if d.get("status") in ("indexed", "failed"):
            return d
        time.sleep(0.2)
    return d


def test_multi_turn_session(client):
    import json as _json
    tok = _token(client, "dave")
    H = {"Authorization": f"Bearer {tok}"}
    kb = client.post("/api/v1/knowledge", json={"name": "mt", "description": ""}, headers=H).json()["id"]
    files = {"file": ("mt.txt", "比亚迪2025年营业收入803.96亿元，电池装机量增长。", "text/plain")}
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H, files=files).json()
    up = _wait_indexed(client, H, up["id"])
    assert up["status"] == "indexed"

    r1 = client.post("/api/v1/chat/stream", headers=H,
                     json={"kb_id": kb, "question": "比亚迪2025年营收多少", "stream": True})
    assert r1.status_code == 200
    sess = None
    for line in r1.text.splitlines():
        if line.startswith("data:"):
            try:
                ev = _json.loads(line[5:].strip())
            except Exception:
                continue
            if ev.get("type") == "sources":
                sess = ev.get("session_id")
    assert sess, "sources 事件应带 session_id"

    r2 = client.post("/api/v1/chat/stream", headers=H,
                     json={"kb_id": kb, "question": "它营收多少", "session_id": sess, "stream": True})
    assert r2.status_code == 200
    assert '"sources"' in r2.text


def test_semantic_cache(client):
    import json as _json
    tok = _token(client, "erin")
    H = {"Authorization": f"Bearer {tok}"}
    kb = client.post("/api/v1/knowledge", json={"name": "cache", "description": ""}, headers=H).json()["id"]
    files = {"file": ("c.txt", "比亚迪2025年营业收入803.96亿元。", "text/plain")}
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H, files=files).json()
    up = _wait_indexed(client, H, up["id"])
    assert up["status"] == "indexed"

    def ask(q):
        r = client.post("/api/v1/chat/stream", headers=H, json={"kb_id": kb, "question": q, "stream": True})
        hit = None
        for line in r.text.splitlines():
            if line.startswith("data:"):
                try:
                    ev = _json.loads(line[5:].strip())
                except Exception:
                    continue
                if ev.get("type") == "done":
                    hit = ev.get("cache_hit")
        return r.status_code, hit

    s1, hit1 = ask("比亚迪营收")
    s2, hit2 = ask("比亚迪营收")  # 完全相同 → 应命中缓存
    assert s1 == 200 and s2 == 200
    assert hit1 is False
    assert hit2 is True
