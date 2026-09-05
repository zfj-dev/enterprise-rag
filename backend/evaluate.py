"""RAG 黄金集评估（论文版）：对运行中的服务跑黄金集，量化"答案含期望事实"率 + "引用页码正确"率。

用法: 先启动 run_real.ps1，再  python backend/evaluate.py
可配环境变量:  EVAL_DOC=backend/paper.pdf  EVAL_GOLDEN=backend/data/golden_set_paper.json
报告写到 logs/eval-report.log（助手可读）。判据：
  - fact_hit: 答案含期望关键字（证明检索到正确内容）
  - page_hit: 表类问题引用了正确页码（证明页码修复生效）
"""
from __future__ import annotations

import json
import os
import time
import traceback

import re

import httpx

BACKEND = os.path.dirname(os.path.abspath(__file__))


def _c(x: str) -> str:
    """去空白（docling 会在数字/标点间加空格，如 '表 4 . 1'、'Windows 11'）。"""
    return re.sub(r"\s+", "", x or "").lower()
BASE = os.environ.get("SELFTEST_BASE", "http://localhost:8000")
REPORT = os.path.join(BACKEND, "logs", "eval-report.log")
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    lines: list[str] = ["=== RAG 黄金集评估报告（论文版）==="]
    lines.append(f"文档: {DOC}")
    results: list[dict] = []
    try:
        c = httpx.Client(base_url=BASE, timeout=300)
        r = c.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        H = {"Authorization": f"Bearer {r.json().get('access_token')}"}
        res = c.post("/api/v1/knowledge", json={"name": "__eval__", "description": ""}, headers=H)
        kb = res.json()["id"]

        with open(DOC, "rb") as f:
            up = c.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                        files={"file": (os.path.basename(DOC), f, "application/pdf")}).json()
        d = {}
        for _ in range(240):
            d = c.get(f"/api/v1/documents/{up['id']}", headers=H).json()
            if d.get("status") in ("indexed", "failed"):
                break
            time.sleep(1)
        lines.append(f"上传: {d.get('status')} chunks={d.get('chunk_count')} 页数={d.get('page_count')}")

        for g in golden:
            q, exp = g["question"], g["expect"]
            exp_page = g.get("page")
            r = c.post("/api/v1/chat/stream", headers=H, json={"kb_id": kb, "question": q, "stream": True})
            body = r.text
            ans, sources = "", []
            for ln in body.splitlines():
                if ln.startswith("data:"):
                    try:
                        ev = json.loads(ln[5:].strip())
                    except Exception:
                        continue
                    if ev.get("type") == "delta":
                        ans += ev.get("text", "")
                    if ev.get("type") == "sources":
                        sources = ev.get("data", [])
            fact_hit = _c(exp) in _c(ans)
            # 忠实度：期望事实是否真的出现在引用来源文本里
            src_text = " ".join(str(s.get("text", "")) for s in sources if isinstance(s, dict))
            grounded = _c(exp) in _c(src_text)
            pages = [s.get("page") for s in sources if isinstance(s, dict)]
            page_hit = (exp_page in pages) if exp_page else None
            results.append({"q": q, "exp": exp, "fact": fact_hit, "page": page_hit, "grounded": grounded, "pages": pages})
            lines.append(f"[{'PASS' if fact_hit else 'FAIL'}] Q:{q} | 期望:{exp} | 命中:{fact_hit}|忠实:{grounded}"
                         + (f" | 页码:{exp_page}->{'✓' if page_hit else pages}" if exp_page else f" | 来源页:{pages}"))
            lines.append(f"    答案前90字: {ans[:90].replace(chr(10),' / ')}")

        c.delete(f"/api/v1/knowledge/{kb}", headers=H)
    except Exception:
        lines.append("fatal: " + traceback.format_exc())

    fact_rate = sum(1 for x in results if x["fact"]) / max(1, len(results))
    grounded_rate = sum(1 for x in results if x["grounded"]) / max(1, len(results))
    page_n = [x for x in results if x["page"] is not None]
    page_rate = (sum(1 for x in page_n if x["page"]) / len(page_n)) if page_n else None
    lines.append(f"\n结果: 答案含期望事实 {round(fact_rate*100)}%  ({sum(1 for x in results if x['fact'])}/{len(results)})")
    lines.append(f"引用忠实度(事实在来源里) {round(grounded_rate*100)}%  ({sum(1 for x in results if x['grounded'])}/{len(results)})")
    if page_rate is not None:
        lines.append(f"引用页码正确 {round(page_rate*100)}%  ({sum(1 for x in page_n if x['page'])}/{len(page_n)})")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
