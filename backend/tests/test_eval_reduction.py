"""压缩降幅进报告（票 19 / #26）：与事实命中**并列**，且没有真实分词器时明确写「不可用」。"""
from __future__ import annotations

from app.eval_core import run_eval

GOLDEN = [{"question": "营收多少", "expect": "803.96", "page": 1}]


def _ask(context):
    def ask(question: str) -> dict:
        return {"answer": "营收803.96亿元。", "sources": [{"text": "营收803.96亿元", "page": 1}],
                "context": context}
    return ask


def _lines(rep) -> list:
    return "\n".join(rep.to_lines()).splitlines()


def test_the_reduction_is_reported_with_its_tokenizer():
    rep = run_eval(GOLDEN, _ask({"tokens_before": 100, "tokens_after": 40,
                                 "tokenizer": "Qwen/Qwen2.5-7B-Instruct"}))
    lines = _lines(rep)
    text = "\n".join(lines)

    assert rep.token_reduction_rate == 0.6
    assert rep.tokenizer_label == "Qwen/Qwen2.5-7B-Instruct"
    assert "上下文压缩降幅(token) 60%" in text
    assert "Qwen/Qwen2.5-7B-Instruct" in text          # 口径就写在数字旁边


def test_the_reduction_sits_right_next_to_the_fact_hit():
    """降幅必须与事实命中**并列** —— 单独报降幅等于奖励「把上下文砍掉」。"""
    lines = _lines(run_eval(GOLDEN, _ask({"tokens_before": 100, "tokens_after": 40,
                                          "tokenizer": "Qwen/x"})))

    fact = next(i for i, ln in enumerate(lines) if ln.startswith("结果: 答案含期望事实"))
    assert lines[fact + 1].startswith("上下文压缩降幅(token)")


def test_without_a_real_tokenizer_the_reduction_is_unavailable_not_estimated():
    """没有真实分词器 -> 写「不可用」+ 原因；**不许**给一个字数估算的百分比。"""
    rep = run_eval(GOLDEN, _ask({"tokens_before": None, "tokens_after": None, "tokenizer": "",
                                 "note": "真实分词器 Qwen/x 不可用：RuntimeError"}))
    lines = _lines(rep)
    row = next(ln for ln in lines if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate is None
    assert "不可用" in row and "RuntimeError" in row
    assert "%" not in row


def test_a_link_that_reports_nothing_says_unavailable_instead_of_faking_a_number():
    """代理链路不装配这份上下文 —— 那就不报数，也不编一个 0%。"""
    rep = run_eval(GOLDEN, lambda q: {"answer": "甲", "sources": []})
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate is None
    assert "不可用" in row


def test_a_real_tokenizer_with_nothing_to_compress_says_not_applicable():
    """分词器好好的、只是这轮没多轮历史 —— 不许把它说成「没有真实分词器」。"""
    rep = run_eval(GOLDEN, _ask({"tokens_before": 0, "tokens_after": 0,
                                 "tokenizer": "Qwen/x", "budget": 3000}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate is None
    assert "不适用" in row and "没有可压的多轮历史" in row and "Qwen/x" in row
    assert "没有真实分词器" not in row


def test_the_scope_line_carries_tokenizer_budget_and_trigger():
    """口径三件套（分词器 / 预算 / 触发条件）要跟数字写在一起（spec 0003 story 22）。"""
    rep = run_eval(GOLDEN, _ask({"tokens_before": 100, "tokens_after": 40,
                                 "tokenizer": "Qwen/x", "budget": 3000}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert "Qwen/x" in row and "预算 3000" in row and "超预算才压" in row


def test_the_sum_and_the_count_use_the_same_filter_as_the_rate():
    """求和 / 条数与降幅必须同一套筛选 —— 否则旁边的数字自己就对不上。"""
    def ask(question):
        return {"answer": "甲", "sources": [], "context": (
            {"tokens_before": 0, "tokens_after": 0, "tokenizer": "Qwen/x"} if question == "b"
            else {"tokens_before": 100, "tokens_after": 40, "tokenizer": "Qwen/x"})}

    rep = run_eval([GOLDEN[0], {**GOLDEN[0], "question": "b"}], ask)
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate == 0.6
    assert "压缩前 100 -> 压缩后 40" in row and "1 条计入" in row
