"""evaluate_rgb.py 的答题函数：**管线给什么就传什么**（#58）。

RGB 段以前只手挑 `answer` / `sources` 两个键，把 `context`（token 口径）与
`citation_coverage`（引用覆盖率）都丢了 —— 报告于是写「不可用」，还把原因
错写成「没有真实分词器」。分词器一直是好的（实测 RGB 进程里是 HfTokenCounter）。
"""
from __future__ import annotations

import evaluate_rgb
from app.services import chat_service

FULL = {
    "session_id": "s", "answer": "甲", "message_id": "m",
    "sources": [{"text": "甲", "page": 1}],
    "context": {"tokens_before": 10, "tokens_after": 4, "tokenizer": "Qwen/x"},
    "citation_coverage": 0.5,
    "trace": {},
}


def test_rgb_answers_pass_the_pipeline_result_through(monkeypatch):
    monkeypatch.setattr(chat_service, "answer", lambda *a, **k: FULL)

    got = evaluate_rgb._answer_fn(db=None, rt=None, user=None)("kb-1", "问题")

    assert got["context"]["tokenizer"] == "Qwen/x"   # token 口径没被丢掉
    assert got["citation_coverage"] == 0.5           # 引用覆盖率没被丢掉
    assert got["answer"] == "甲" and got["sources"]


def test_the_passthrough_does_not_invent_missing_fields(monkeypatch):
    """管线没给的键就不该凭空长出来 —— 透传不等于编字段。"""
    monkeypatch.setattr(chat_service, "answer",
                        lambda *a, **k: {"answer": "乙", "sources": []})

    got = evaluate_rgb._answer_fn(db=None, rt=None, user=None)("kb-1", "问题")

    assert got == {"answer": "乙", "sources": []}
