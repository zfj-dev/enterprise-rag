"""端到端自检：对运行中的服务(localhost:8000)跑完整链路，写报告到 logs/selftest-report.log。

用法: 先启动 run_real.ps1，再运行  scripts/selftest.ps1（或 python selftest.py）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import httpx

BASE = os.environ.get("SELFTEST_BASE", "http://localhost:8000")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "selftest-report.log")
ADMIN = ("admin", "admin123")

results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    H: dict = {}
    try:
        c = httpx.Client(base_url=BASE, timeout=180)

        # 1) health
        try:
            r = c.get("/health")
            js = r.json()
            step("health", r.status_code == 200 and js.get("status") == "ok", f"use_real={js.get('use_real')}")
        except Exception as e:
            step("health", False, f"服务没起来? {e}")
            _write_report()
            sys.exit(1)

        # 2) login
        try:
            r = c.post("/api/v1/auth/login", json={"username": ADMIN[0], "password": ADMIN[1]})
            tok = r.json().get("access_token")
            H = {"Authorization": f"Bearer {tok}"}
            step("login", r.status_code == 200 and bool(tok), f"{r.status_code}")
        except Exception as e:
            step("login", False, str(e))
            _write_report()
            return

        # 3) create KB
        try:
            r = c.post("/api/v1/knowledge", json={"name": "__selftest__", "description": ""}, headers=H)
            kb_id = r.json().get("id")
            step("create_kb", r.status_code == 200 and bool(kb_id), f"{r.status_code}")
        except Exception as e:
            step("create_kb", False, str(e))
            _write_report()
            return

        # 4) upload (async) + wait indexed
        doc_id = None
        try:
            files = {"file": ("selftest.txt",
                              "比亚迪2025年营业收入803.96亿元，同比增长35%，电池装机量持续增长，海外销量翻倍。",
                              "text/plain")}
            r = c.post(f"/api/v1/documents?kb_id={kb_id}", headers=H, files=files)
            doc = r.json()
            doc_id = doc.get("id")
            step("upload", r.status_code == 200 and doc.get("status") in ("processing", "pending"), f"{r.status_code} {doc.get('status')}")
        except Exception as e:
            step("upload", False, str(e))
            _write_report()
            return

        try:
            d = {}
            for _ in range(120):
                d = c.get(f"/api/v1/documents/{doc_id}", headers=H).json()
                if d.get("status") in ("indexed", "failed"):
                    break
                time.sleep(1)
            step("indexed", d.get("status") == "indexed",
                 f"status={d.get('status')} chunks={d.get('chunk_count')} err={d.get('error')}")
        except Exception as e:
            step("indexed", False, str(e))
            _write_report()
            return

        # 5) chat stream
        msg_id = None
        answer_snippet = ""
        try:
            r = c.post("/api/v1/chat/stream", headers=H,
                       json={"kb_id": kb_id, "question": "比亚迪25年营业收入是多少", "stream": True})
            body = r.text
            fake = ("模拟回答" in body) or ("未接云端" in body)
            has_sources = '"sources"' in body
            for line in body.splitlines():
                if line.startswith("data:"):
                    try:
                        ev = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    if ev.get("type") == "delta":
                        answer_snippet = (answer_snippet + ev.get("text", ""))[:200]
                    if ev.get("type") == "done":
                        msg_id = ev.get("message_id")
            step("chat_status", r.status_code == 200, f"{r.status_code}")
            step("chat_real_llm", not fake, f"fake回答={fake}")
            step("chat_sources", has_sources, f"has_sources={has_sources}")
            step("chat_done", bool(msg_id), f"msg_id={bool(msg_id)}")
            step("chat_answer_preview", len(answer_snippet) > 10, f"前200字: {answer_snippet}")
        except Exception as e:
            step("chat", False, str(e))

        # 6) feedback
        try:
            if msg_id:
                r = c.post("/api/v1/feedback", json={"message_id": msg_id, "rating": 1}, headers=H)
                step("feedback", r.status_code == 200, f"{r.status_code}")
            else:
                step("feedback", False, "无 message_id")
        except Exception as e:
            step("feedback", False, str(e))

        # 7) debug
        try:
            r = c.get(f"/api/v1/debug/query?kb_id={kb_id}&question=营收", headers=H)
            step("debug", r.status_code == 200 and "retrieval_top" in r.json(), f"{r.status_code}")
        except Exception as e:
            step("debug", False, str(e))

        # 8) cleanup
        try:
            c.delete(f"/api/v1/knowledge/{kb_id}", headers=H)
            step("cleanup", True, "")
        except Exception:
            pass

        _write_report()
    except Exception:
        step("fatal", False, traceback.format_exc())
        _write_report()


def _write_report() -> None:
    passed = sum(1 for _, ok, _ in results if ok)
    lines = ["=== RAG 端到端自检报告 ==="]
    for name, ok, detail in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    lines.append("")
    lines.append(f"结果: {passed}/{len(results)} passed")
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n报告已写入: {REPORT}")
    except Exception as e:
        print(f"\n写报告失败: {e}")
    print(f"结果: {passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
