# -*- coding: utf-8 -*-
"""Step2 后端API边界扫描：遍历路由发异常请求，检查500/数据不合理/是否拒绝；并发50次；双用户越权。"""
import httpx, json, os, re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

BASE = "http://localhost:8000/api/v1"
SPECIAL = "<>\"'&%|$;\\"
LONG = "x" * 5000
LONGBIG = "A" * 100000
ANOM = []


def anom(kind, case, desc):
    ANOM.append({"kind": kind, "case": case, "desc": desc})
    print(f"  [异常][{kind}] {case}: {desc}")


def login(u, p):
    r = httpx.post(f"{BASE}/auth/login", json={"username": u, "password": p}, timeout=20)
    return r.json().get("access_token")


def req(case, method, path, token=None, json_body=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        t = {"GET": httpx.get, "POST": httpx.post, "PATCH": httpx.patch, "DELETE": httpx.delete}[method]
        kw = {"headers": h, "timeout": 25}
        if method in ("POST", "PATCH") and json_body is not None:
            kw["json"] = json_body
        return t(BASE + path, **kw)
    except Exception as e:
        anom("conn", f"{method} {path}", f"请求异常 {type(e).__name__}: {e}")
        return None


def flag(r, case):
    if r is None:
        return
    if r.status_code >= 500:
        anom("500", case, f"收到 {r.status_code}: {r.text[:120]}")
    return r


def main():
    token = login("admin", "admin123")
    if not token:
        print("无 token，退出")
        return
    kb = httpx.post(f"{BASE}/knowledge", headers={"Authorization": f"Bearer {token}"},
                    json={"name": "扫描库", "description": ""}, timeout=20).json()
    kb_id = kb.get("id")
    print(f"KB: {kb_id}")

    cases = [
        ("login_empty", "POST", "/auth/login", {"username": "", "password": ""}),
        ("login_null", "POST", "/auth/login", {"username": None, "password": None}),
        ("login_overlong", "POST", "/auth/login", {"username": LONG, "password": LONG}),
        ("login_special", "POST", "/auth/login", {"username": SPECIAL, "password": SPECIAL}),
        ("login_miss_field", "POST", "/auth/login", {"username": "admin"}),
        ("register_empty", "POST", "/auth/register", {"username": "", "password": ""}),
        ("kb_create_empty", "POST", "/knowledge", {}),
        ("kb_create_blank", "POST", "/knowledge", {"name": "", "description": ""}),
        ("kb_create_overlong", "POST", "/knowledge", {"name": LONG, "description": ""}),
        ("kb_patch_bad", "PATCH", "/knowledge/bad-id", {"name": "x"}),
        ("kb_patch_type", "PATCH", f"/knowledge/{kb_id}", {"name": 12345}),
        ("kb_delete_bad", "DELETE", "/knowledge/bad-id", None),
        ("docs_no_kb", "GET", "/documents?kb_id=", None),
        ("docs_bad_kb", "GET", "/documents?kb_id=bad", None),
        ("docs_csv", "GET", "/documents?kb_id=" + (kb_id or ""), None),
        ("doc_get_bad", "GET", "/documents/bad-id", None),
        ("doc_content_bad", "GET", "/documents/bad-id/content", None),
        ("doc_file_bad", "GET", "/documents/bad-id/file", None),
        ("chat_no_kb", "GET", "/chat/sessions?kb_id=", None),
        ("chat_bad_kb", "GET", "/chat/sessions?kb_id=bad", None),
        ("chat_sess_bad", "GET", "/chat/sessions/bad-id", None),
        ("feedback_empty", "POST", "/feedback", {}),
        ("debug_no_param", "GET", "/debug/query", None),
    ]
    print("=== 异常输入 ===")
    stat_lines = []
    for case, m, path, body in cases:
        r = req(case, m, path, token, body)
        flag(r, case)
        if r is not None:
            stat_lines.append(f"  {case}: {r.status_code}")
    print("\n".join(stat_lines))

    print("=== chat/stream 异常 body ===")
    for case, body in [("stream_empty", {}), ("stream_no_kb", {"question": "hi"}),
                       ("stream_long", {"kb_id": kb_id or "x", "question": LONGBIG}),
                       ("stream_null", {"kb_id": None, "question": None}),
                       ("stream_special", {"kb_id": kb_id or "x", "question": SPECIAL})]:
        try:
            with httpx.stream("POST", f"{BASE}/chat/stream",
                              headers={"Authorization": f"Bearer {token}"}, json=body, timeout=15) as resp:
                if resp.status_code >= 500:
                    anom("500", case, f"{resp.status_code}")
                print(f"  {case}: {resp.status_code}")
        except Exception as e:
            anom("stream", case, f"{type(e).__name__}: {e}")

    print("=== 并发 50x GET /documents ===")
    def conq(_):
        try:
            r = httpx.get(f"{BASE}/documents?kb_id={kb_id}", headers={"Authorization": f"Bearer {token}"}, timeout=30)
            return r.status_code
        except Exception as e:
            return "EXC:" + str(e)
    with ThreadPoolExecutor(max_workers=20) as ex:
        codes = list(ex.map(conq, range(50)))
    cc = Counter(codes)
    print("  并发状态码分布:", dict(cc))

    print("=== 并发 50x SSE /chat/stream ===")
    def ss(_):
        try:
            with httpx.stream("POST", f"{BASE}/chat/stream",
                              headers={"Authorization": f"Bearer {token}"},
                              json={"kb_id": kb_id, "question": "并发测试"}, timeout=20) as resp:
                return resp.status_code
        except Exception:
            return "EXC"
    with ThreadPoolExecutor(max_workers=20) as ex:
        codes2 = list(ex.map(ss, range(50)))
    cc2 = Counter(codes2)
    print("  SSE 并发状态码分布:", dict(cc2))
    if cc.get(500) or cc2.get(500) or any(isinstance(k, str) for k in cc) or cc2.get("EXC"):
        anom("concurrency", "doc/stream", f"GET分布={dict(cc)} SSE分布={dict(cc2)}")

    print("=== 越权（双用户） ===")
    u2 = "scanuser2"
    httpx.post(f"{BASE}/auth/register", json={"username": u2, "password": "pass12345"}, timeout=15)
    tok2 = login(u2, "pass12345")
    sess_id = None
    try:
        with httpx.stream("POST", f"{BASE}/chat/stream",
                          headers={"Authorization": f"Bearer {token}"},
                          json={"kb_id": kb_id, "question": "越权测试", "session_id": "00000000-0000-0000-0000-0000000000ab"}, timeout=15) as resp:
            for line in resp.iter_lines():
                m = re.search(r'"session_id":\s*"([^"]+)"', line)
                if m:
                    sess_id = m.group(1)
                    break
    except Exception as e:
        print("  会话创建异常", e)
    print(f"  admin 会话: {sess_id}")
    if sess_id:
        r = httpx.get(f"{BASE}/chat/sessions/{sess_id}", headers={"Authorization": f"Bearer {tok2}"}, timeout=15)
        if r.status_code != 404:
            anom("越权", "user2 读 admin 会话", f"返回 {r.status_code}（期望404）")
        print(f"  user2 GET 会话: {r.status_code}（期望404）")
        try:
            with httpx.stream("POST", f"{BASE}/chat/stream",
                              headers={"Authorization": f"Bearer {tok2}"},
                              json={"kb_id": kb_id, "question": "越权", "session_id": sess_id}, timeout=15) as resp:
                print(f"  user2 用 admin session 流式: {resp.status_code}（期望404）")
                if resp.status_code != 404:
                    anom("越权", "user2 用admin session流式", f"返回 {resp.status_code}")
        except Exception as e:
            anom("越权", "user2 用admin session流式", f"异常 {e}")

    with open(os.path.join(os.path.dirname(__file__), "api_scan_report.json"), "w", encoding="utf-8") as f:
        json.dump({"anomalies": ANOM}, f, ensure_ascii=False, indent=2)
    print(f"\n===== API 扫描完成：异常 {len(ANOM)} 条 =====")
    for a in ANOM:
        print(f"  [{a['kind']}] {a['case']}: {a['desc']}")


if __name__ == "__main__":
    main()
