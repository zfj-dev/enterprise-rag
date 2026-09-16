"""端到端 API 集成测试：注册→登录→建库→上传文档→入库→流式问答→反馈→调试。"""
from __future__ import annotations

from tests.helpers import sse_events


def _token(client, username: str) -> str:
    r = client.post("/api/v1/auth/register", json={"username": username, "password": "pw123456"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_health(client):
    r = client.get("/api/v1/auth/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------- 越权与文件隔离（#64）----------

def _kb_for(client, H, name="K"):
    r = client.post("/api/v1/knowledge", json={"name": name}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_two_users_uploading_the_same_filename_do_not_share_a_file(client):
    """同名文件不许共用一个磁盘路径 —— 一方覆盖另一方、还把别人的字节发出去（#64）。

    老代码按文件名存进一个全局目录，`/documents/{id}/file` 照着 file_path 直接发字节，
    所以两个用户传同名文件 = 互相看到对方的文件。
    """
    HA = {"Authorization": f"Bearer {_token(client, 'same_a')}"}
    HB = {"Authorization": f"Bearer {_token(client, 'same_b')}"}
    ka, kb = _kb_for(client, HA, "A"), _kb_for(client, HB, "B")

    da = client.post(f"/api/v1/documents?kb_id={ka}", headers=HA,
                     files={"file": ("报表.txt", "ALICE_SECRET", "text/plain")}).json()
    client.post(f"/api/v1/documents?kb_id={kb}", headers=HB,
                files={"file": ("报表.txt", "BOB_SECRET", "text/plain")})

    got = client.get(f"/api/v1/documents/{da['id']}/file", headers=HA)
    assert got.status_code == 200
    assert got.content == b"ALICE_SECRET"          # 不是 BOB_SECRET


def test_skipping_a_same_name_upload_does_not_touch_the_stored_file(client):
    """`overwrite=false` 走「跳过」分支时，盘上的原文件一个字节都不该动（#64）。"""
    H = {"Authorization": f"Bearer {_token(client, 'skip_user')}"}
    kb = _kb_for(client, H)
    first = client.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                        files={"file": ("同名.txt", "原始内容", "text/plain")}).json()

    again = client.post(f"/api/v1/documents?kb_id={kb}&overwrite=false", headers=H,
                        files={"file": ("同名.txt", "覆盖内容", "text/plain")}).json()

    assert again["id"] == first["id"]              # 确实走了跳过分支
    got = client.get(f"/api/v1/documents/{first['id']}/file", headers=H)
    assert got.content.decode("utf-8") == "原始内容"


def test_feedback_cannot_be_written_onto_someone_elses_message(client):
    """反馈只能打在自己会话的消息上 —— 只查「消息存在」是越权（#64）。"""
    HA = {"Authorization": f"Bearer {_token(client, 'fb_a')}"}
    HB = {"Authorization": f"Bearer {_token(client, 'fb_b')}"}
    kb = _kb_for(client, HA)
    client.post(f"/api/v1/documents?kb_id={kb}", headers=HA,
                files={"file": ("a.txt", "比亚迪2025年营业收入为803.96亿元。", "text/plain")})
    r = client.post("/api/v1/chat/stream", headers=HA,
                    json={"kb_id": kb, "question": "营收多少", "stream": True})
    mid = [e for e in sse_events(r.text) if e.get("type") == "done"][-1]["message_id"]

    assert client.post("/api/v1/feedback", headers=HB,
                       json={"message_id": mid, "rating": -1}).status_code == 404
    assert client.post("/api/v1/feedback", headers=HA,
                       json={"message_id": mid, "rating": 9999}).status_code == 400
    assert client.post("/api/v1/feedback", headers=HA,
                       json={"message_id": mid, "rating": 1}).status_code == 200


def test_chatting_against_someone_elses_knowledge_base_is_not_found(client):
    """拿别人的 kb_id 提问一律 404 —— kb_id 会进语义缓存的键，不隔离就是泄漏（#64）。"""
    HA = {"Authorization": f"Bearer {_token(client, 'kb_a')}"}
    HB = {"Authorization": f"Bearer {_token(client, 'kb_b')}"}
    kb = _kb_for(client, HA)

    r = client.post("/api/v1/chat/stream", headers=HB,
                    json={"kb_id": kb, "question": "随便问问", "stream": True})
    assert r.status_code == 404


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
    dones = [e for e in sse_events(cr.text) if e.get("type") == "done"]
    msg_id = dones[-1].get("message_id") if dones else None
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
    sess = next((e.get("session_id") for e in sse_events(r1.text) if e.get("type") == "sources"), None)
    assert sess, "sources 事件应带 session_id"

    r2 = client.post("/api/v1/chat/stream", headers=H,
                     json={"kb_id": kb, "question": "它营收多少", "session_id": sess, "stream": True})
    assert r2.status_code == 200
    assert '"sources"' in r2.text


def test_semantic_cache(client):
    tok = _token(client, "erin")
    H = {"Authorization": f"Bearer {tok}"}
    kb = client.post("/api/v1/knowledge", json={"name": "cache", "description": ""}, headers=H).json()["id"]
    files = {"file": ("c.txt", "比亚迪2025年营业收入803.96亿元。", "text/plain")}
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H, files=files).json()
    up = _wait_indexed(client, H, up["id"])
    assert up["status"] == "indexed"

    def ask(q):
        r = client.post("/api/v1/chat/stream", headers=H, json={"kb_id": kb, "question": q, "stream": True})
        hits = [e.get("cache_hit") for e in sse_events(r.text) if e.get("type") == "done"]
        return r.status_code, (hits[-1] if hits else None)

    s1, hit1 = ask("比亚迪营收")
    s2, hit2 = ask("比亚迪营收")  # 完全相同 → 应命中缓存
    assert s1 == 200 and s2 == 200
    assert hit1 is False
    assert hit2 is True
