"""黄金集评估：对运行中的服务(localhost:8000)跑黄金集，算"答案含期望事实"率与检索来源。

用法: 先启动 run_real.ps1，再  python backend/evaluate.py（或 scripts/selftest.ps1 同模式）。
报告写到 logs/eval-report.log（助手可直接读）。
"""
from __future__ import annotations

import json
import os
import time
import traceback

import httpx

BASE = os.environ.get("SELFTEST_BASE", "http://localhost:8000")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "eval-report.log")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden_set.json")
SAMPLE_DOC = (
    "比亚迪2025年营业收入803.96亿元，同比增长35%。动力电池装机量增长40%，"
    "海外销量翻倍，研发投入占比6%。公司加快全球化布局。"
)


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    c = httpx.Client(base_url=BASE, timeout=180)
    lines: list[str] = ["=== RAG 黄金集评估报告 ==="]
    results: list[tuple[str, bool, bool]] = []  # (q, answer_hit, has_sources)
    try:
        r = c.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        H = {"Authorization": f"Bearer {r.json().get('access_token')}"}

        kb = c.post("/api/v1/knowledge", json={"name": "__eval__", "description": ""}, headers=H).json()["id"]
        up = c.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                    files={"file": ("eval.txt", SAMPLE_DOC, "text/plain")}).json()
        d = {}
        for _ in range(120):
            d = c.get(f"/api/v1/documents/{up['id']}", headers=H).json()
            if d.get("status") in ("indexed", "failed"):
                break
            time.sleep(1)
        lines.append(f"样本文档: {d.get('status')} chunks={d.get('chunk_count')}")

        for g in golden:
            q, exp = g["question"], g["expect"]
            r = c.post("/api/v1/chat/stream", headers=H,
                       json={"kb_id": kb, "question": q, "stream": True})
            body = r.text
            ans, has_src = "", '"sources"' in body
            for ln in body.splitlines():
                if ln.startswith("data:"):
                    try:
                        ev = json.loads(ln[5:].strip())
                    except Exception:
                        continue
                    if ev.get("type") == "delta":
                        ans += ev.get("text", "")
            hit = exp.lower() in ans.lower()
            results.append((q, hit, has_src))
            lines.append(f"[{'PASS' if hit else 'FAIL'}] Q:{q} | 期望含:{exp} | 命中:{hit} | 来源:{has_src}")
            lines.append(f"    答案前80字: {ans[:80]}")

        c.delete(f"/api/v1/knowledge/{kb}", headers=H)
    except Exception:
        lines.append("fatal: " + traceback.format_exc())

    passed = sum(1 for _, hit, _ in results if hit)
    src_ok = sum(1 for _, _, has in results if has)
    lines.append("")
    lines.append(f"结果: 答案含期望事实 {passed}/{len(results)}；带来源 {src_ok}/{len(results)}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
