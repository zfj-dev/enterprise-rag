"""evaluate.py 的职责边界（票 01 / #8）：只做 HTTP 管线与落盘，判据一律交给评测核心。

用假的 httpx.Client 照剧本应答，整条脚本离线跑通。
"""
from __future__ import annotations

import json

import evaluate


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    """只会照剧本应答的假 httpx.Client。"""

    def __init__(self, *args, **kwargs):
        pass

    def post(self, url, headers=None, json=None, files=None):
        if url.endswith("/auth/login"):
            return _Resp({"access_token": "t"})
        if url.endswith("/chat/stream"):
            return _Resp(text='data: {"type": "delta", "text": "甲"}')
        if url.startswith("/api/v1/knowledge"):
            return _Resp({"id": "kb1"})
        if url.startswith("/api/v1/documents"):
            return _Resp({"id": "doc1"})
        return _Resp({})

    def get(self, url, headers=None):
        return _Resp({"status": "indexed", "chunk_count": 1, "page_count": 1})

    def delete(self, url, headers=None):
        return _Resp({"ok": True})


# ---------- SSE 解析 ----------

def test_parse_sse_collects_answer_and_sources():
    body = "\n".join([
        'data: {"type": "sources", "data": [{"text": "甲", "page": 1}]}',
        "",
        'data: {"type": "delta", "text": "答"}',
        'data: {"type": "delta", "text": "案"}',
        "",
        'data: {"type": "done"}',
    ])
    assert evaluate._parse_sse(body) == {"answer": "答案", "sources": [{"text": "甲", "page": 1}]}


def test_parse_sse_ignores_malformed_and_non_data_lines():
    assert evaluate._parse_sse("data: not-json\n\n: keep-alive\n") == {"answer": "", "sources": []}


def test_answer_fn_asks_the_running_service():
    class C:
        def __init__(self):
            self.calls = []

        def post(self, url, headers=None, json=None):
            self.calls.append((url, json))
            return _Resp(text='data: {"type": "delta", "text": "甲"}')

    c = C()
    assert evaluate._answer_fn(c, "kb1", {})("问")["answer"] == "甲"
    assert c.calls == [("/api/v1/chat/stream", {"kb_id": "kb1", "question": "问", "stream": True})]


# ---------- 整条脚本：判据来自核心 ----------

def test_main_runs_offline_and_judges_via_the_core(tmp_path, monkeypatch):
    """离线跑通 main()：脚本自己没有判据，期望事实命中一律由核心算出。"""
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "Q", "expect": "甲"}]), encoding="utf-8")
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4")
    report = tmp_path / "eval-report.log"

    monkeypatch.setattr(evaluate, "GOLDEN", str(golden))
    monkeypatch.setattr(evaluate, "DOC", str(doc))
    monkeypatch.setattr(evaluate, "REPORT", str(report))
    monkeypatch.setattr(evaluate.httpx, "Client", _FakeClient)

    seen = {}
    real_run_eval = evaluate.run_eval

    def spy(goldenset, answer_fn, judge_fn=None, judge_label=None):
        seen["golden"] = list(goldenset)
        seen["answer"] = answer_fn("Q")
        seen["judge_label"] = judge_label
        return real_run_eval(goldenset, answer_fn, judge_fn, judge_label)

    monkeypatch.setattr(evaluate, "run_eval", spy)
    evaluate.main()

    assert seen["golden"] == [{"question": "Q", "expect": "甲"}]
    assert seen["answer"]["answer"] == "甲"          # answer_fn 真的打到服务（假 client）

    text = report.read_text(encoding="utf-8")
    assert "答案含期望事实 100%" in text
    assert "口径" in text                             # 报告带口径
    assert "上传: indexed" in text                     # 头部信息也在
    assert "=== RAGAS 四项 ===" in text                 # RAGAS 段无论有没有裁判都要出
