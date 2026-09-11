"""RAG 黄金集评估：对运行中的服务跑黄金集，出「答案含期望事实」等数字。

判据本身在可注入的评测核心 app/eval_core.py；本脚本只做两件事：
把运行中的服务包成 answer_fn、把核心给的报告落盘。

用法: 先启动 run_real.ps1，再  python backend/evaluate.py
可配环境变量:  EVAL_DOC=backend/paper.pdf  EVAL_GOLDEN=backend/data/golden_set_paper.json
报告写到 logs/eval-report.log（助手可读）。

黄金集条目（JSON 数组，每条一个对象）：
  question  问题
  expect    期望事实（判 fact_hit / grounded 用）
  page      可选，期望页码（判 page_hit 用）
  negative  可选，true = 负样本：答案不在文档里，期望系统拒答；
            这种条目不写 expect，也不参与事实 / 页码的分母，只计入拒答率
"""
from __future__ import annotations

import json
import os
import time
import traceback

import httpx

from app.eval_core import run_eval

BACKEND = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("SELFTEST_BASE", "http://localhost:8000")
REPORT = os.path.join(BACKEND, "logs", "eval-report.log")
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))


def _parse_sse(body: str) -> dict:
    """把 /chat/stream 的 SSE 响应体拼成核心要的 {answer, sources}。"""
    answer, sources = "", []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except Exception:
            continue
        if ev.get("type") == "delta":
            answer += ev.get("text", "")
        elif ev.get("type") == "sources":
            sources = ev.get("data", [])
    return {"answer": answer, "sources": sources}


def _answer_fn(client: httpx.Client, kb_id: str, headers: dict):
    """把运行中的服务包成评测核心要的 answer_fn（问题 -> 答案 + 来源）。"""
    def ask(question: str) -> dict:
        r = client.post("/api/v1/chat/stream", headers=headers,
                        json={"kb_id": kb_id, "question": question, "stream": True})
        return _parse_sse(r.text)
    return ask


def _upload(client: httpx.Client, kb_id: str, headers: dict) -> str:
    """上传被评文档并等入库，返回一行状态描述。"""
    with open(DOC, "rb") as f:
        up = client.post("/api/v1/documents?kb_id=%s" % kb_id, headers=headers,
                         files={"file": (os.path.basename(DOC), f, "application/pdf")}).json()
    d = {}
    for _ in range(240):
        d = client.get("/api/v1/documents/%s" % up["id"], headers=headers).json()
        if d.get("status") in ("indexed", "failed"):
            break
        time.sleep(1)
    return "上传: %s chunks=%s 页数=%s" % (d.get("status"), d.get("chunk_count"), d.get("page_count"))


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    head = ["=== RAG 黄金集评估报告 ===",
            "黄金集: %s" % GOLDEN, "被评文档: %s" % DOC]
    try:
        c = httpx.Client(base_url=BASE, timeout=300)
        r = c.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        H = {"Authorization": "Bearer %s" % r.json().get("access_token")}
        kb = c.post("/api/v1/knowledge", json={"name": "__eval__", "description": ""},
                    headers=H).json()["id"]
        head.append(_upload(c, kb, H))
        head.append("")
        body = run_eval(golden, _answer_fn(c, kb, H)).to_lines()
        c.delete("/api/v1/knowledge/%s" % kb, headers=H)
    except Exception:
        body = ["fatal: " + traceback.format_exc()]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
