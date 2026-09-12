"""延迟评测：并发下的分段耗时（检索 / 重排 / 生成 / 首字）与 P50 / P95。

分段口径来自服务端 done 事件里的 `latency` 字段（管线自己记的毫秒数：检索与重排由检索器
计时，生成与首字由问答管线计时）；TTFT 另由客户端测「请求发出 → 第一个 delta」相互印证。

**本脚本只测延迟，不判正确性** —— 正确性由 evaluate.py 那一次单次顺序跑判（票 02）。
并发下再判对错会把两件事混在一起，所以刻意分开。

用法: 先启动服务，再  python backend/evaluate_latency.py
报告: logs/latency-report.log
环境:  EVAL_CONCURRENCY 并发数（默认 4）；EVAL_ROUNDS 轮数（默认 2）；
       EVAL_GOLDEN 取题用的黄金集；EVAL_DOC / SELFTEST_BASE 同 evaluate.py

⚠️ 服务端有**每用户**同时流式上限（config: max_concurrent_streams_per_user，默认 2）。
   要测 3–5 并发，请把服务端该配置调高（如 8）再重启 —— 否则会被 429 挡住，
   报告里会明说「被并发上限挡住」，不会拿残缺样本充数。
"""
from __future__ import annotations

import json
import os
import threading
import time

import httpx

from app.eval_core import LATENCY_STAGES, LatencyMetrics
from app.eval_http import upload_and_wait

BACKEND = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("SELFTEST_BASE", "http://localhost:8000")
REPORT = os.path.join(BACKEND, "logs", "latency-report.log")
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))
ROUNDS = int(os.environ.get("EVAL_ROUNDS", "2"))

NOTE = (
    "并发 N 个请求同时打 /chat/stream（N=%d），每轮取 N 道题。"
    "检索 / 重排 / 生成三段取服务端 done 事件的 latency 字段（管线自己记的毫秒数）："
    "检索 = 嵌入 + 向量 + BM25 + RRF；重排 = 重排模型那一步；"
    "生成 = 开始生成到最后一个增量（**引用校验与落库不计入任何一段**，它们不是这三步）。"
    "首字(TTFT) 由客户端测：从请求发出到第一个增量，**含检索与重排** —— 这才是用户感知到的首字时间"
) % CONCURRENCY


def _one(client: httpx.Client, headers: dict, kb_id: str, question: str) -> dict:
    """发一次问答，回收这一条的分段耗时（毫秒）。被 429 挡下则标 status。"""
    t0 = time.perf_counter()
    ttft = None
    latency: dict = {}
    rerank: dict = {}       # 这次的重排降级没降级（票 39 / #48）
    status = 200
    with client.stream("POST", "/api/v1/chat/stream", headers=headers,
                       json={"kb_id": kb_id, "question": question, "stream": True}) as r:
        status = r.status_code
        if status != 200:
            return {"status": status}
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if ev.get("type") == "delta" and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            elif ev.get("type") == "done":
                latency = ev.get("latency") or {}
                rerank = ev.get("rerank") or {}
    # 分段名只认核心的 LATENCY_STAGES（线上字段名统一是 <段名>_ms），别在这里再抄一份
    sample = {stage: latency.get(stage + "_ms") for stage in LATENCY_STAGES}
    sample["ttft"] = ttft if ttft is not None else latency.get("ttft_ms")
    # 降级时下面那个「重排」耗时是**因为没重排**才有的数字 —— 要能标出来（票 39 / #48）
    sample["rerank_degraded"] = rerank.get("degraded")
    sample["status"] = status
    return sample


def _run_round(client, headers, kb_id, questions: list) -> list:
    """一轮：N 道题同时发出去。"""
    out: list = []
    lock = threading.Lock()

    def worker(q: str) -> None:
        try:
            got = _one(client, headers, kb_id, q)
        except Exception as e:   # noqa: BLE001 —— 单条挂了不该拖垮整轮，记下来即可
            got = {"status": "error", "error": "%s: %s" % (type(e).__name__, e)}
        with lock:
            out.append(got)

    threads = [threading.Thread(target=worker, args=(q,)) for q in questions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def _upload(client, kb_id: str, headers: dict) -> str:
    d = upload_and_wait(client, kb_id, headers, DOC)
    return "上传: %s chunks=%s" % (d.get("status"), d.get("chunk_count"))


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    questions = [g["question"] for g in golden if not g.get("negative")] or ["测试问题"]

    lines = ["=== 延迟评测 ===", "文档: %s" % DOC, "并发: %d × %d 轮" % (CONCURRENCY, ROUNDS)]
    samples: list = []
    try:
        c = httpx.Client(base_url=BASE, timeout=300)
        r = c.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        H = {"Authorization": "Bearer %s" % r.json().get("access_token")}
        kb = c.post("/api/v1/knowledge", json={"name": "__latency__", "description": ""},
                    headers=H).json()["id"]
        lines.append(_upload(c, kb, H))
        lines.append("")

        for _ in range(ROUNDS):
            batch = (questions * ((CONCURRENCY // len(questions)) + 1))[:CONCURRENCY]
            samples.extend(_run_round(c, H, kb, batch))
        c.delete("/api/v1/knowledge/%s" % kb, headers=H)
    except Exception as e:   # noqa: BLE001 —— 这页是给人看的，写一行就够；traceback 太长会淹掉其它段
        lines.append("未跑：%s: %s" % (type(e).__name__, e))

    blocked = [s for s in samples if s.get("status") == 429]
    ok = [s for s in samples if s.get("status") == 200 and s.get("generate") is not None]
    if blocked:
        lines += ["被每用户并发上限挡住了 %d 条（HTTP 429）—— 本次数字不完整。" % len(blocked),
                  "请把服务端 max_concurrent_streams_per_user 调高（如 8）后重启再跑。", ""]

    degraded = [s for s in ok if s.get("rerank_degraded") is True]
    if degraded:
        lines += ["**有 %d/%d 条请求的重排降级为 RRF 原顺序** —— 下面那一行「重排」耗时"
                  "是没重排时的数字，不能当成重排性能看。" % (len(degraded), len(ok)), ""]

    metrics = LatencyMetrics(concurrent=CONCURRENCY, rounds=ROUNDS, samples=ok, note=NOTE)
    lines.extend(metrics.to_lines())
    if len(ok) != len(samples):
        lines.append("（共 %d 条请求，其中 %d 条拿到完整分段）" % (len(samples), len(ok)))

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
