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


def _build_judge():
    """按固定口径造 RAGAS 裁判（judge 模型 + 温度 0）。

    返回 (裁判, 要不到的原因)：要不到时裁判为 None、原因原样返回 —— 报告里会写明，
    **绝不给看着正常的假数字**。只有真异常才往外抛。
    """
    from app.config import get_settings
    from app.core.embedding import get_embedding
    from app.core.llm import CloudLLM
    from app.eval_judge import JudgeUnavailable, RagasJudge

    s = get_settings()
    try:
        if s.llm_provider == "fake":
            raise JudgeUnavailable("LLM_PROVIDER=fake（演示/假模型），没有真实裁判可用")
        llm = CloudLLM(model=s.ragas_judge_model, temperature=s.ragas_judge_temperature)
        if not (llm.api_key or ""):
            raise JudgeUnavailable("未配置 LLM API Key，RAGAS 裁判不可用")   # 先查 Key，别白加载嵌入
        return RagasJudge(llm, get_embedding()), None
    except JudgeUnavailable as e:
        return None, str(e)
    except Exception as e:   # noqa: BLE001 —— 嵌入/模型加载失败也算裁判要不到，但要说清是哪一类
        return None, "%s: %s" % (type(e).__name__, e)


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

        judge, judge_note = _build_judge()
        report = run_eval(golden, _answer_fn(c, kb, H),
                          judge_fn=judge, judge_label=judge.label if judge else None)
        if judge_note:
            report.judge_error = report.judge_error or judge_note
        body = report.to_lines()
        c.delete("/api/v1/knowledge/%s" % kb, headers=H)
    except Exception:
        body = ["fatal: " + traceback.format_exc()]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
