"""压缩降幅进报告（票 19 / #26）：与事实命中**并列**，且没有真实分词器时明确写「不可用」。"""
from __future__ import annotations

from app.eval_core import run_eval

GOLDEN = [{"question": "营收多少", "expect": "803.96", "page": 1}]


def _ask(context):
    def ask(question: str) -> dict:
        return {"answer": "营收803.96亿元。", "sources": [{"text": "营收803.96亿元", "page": 1}],
                "context": context}
    return ask


def mix(first: dict, second: dict):
    """两条条目各给一份 context：第一条是 GOLDEN[0]，第二条是问题 "b"。

    两份都补上 `tokenizer`，免得「口径是谁」意外变成被测点。
    """
    def ask(question: str) -> dict:
        ctx = dict(second if question == "b" else first)
        ctx.setdefault("tokenizer", "Qwen/x")
        return {"answer": "甲", "sources": [], "context": ctx}
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

    rep = run_eval([GOLDEN[0], {**GOLDEN[0], "question": "b"}], mix(
        {"tokens_before": 100, "tokens_after": 40},
        {"tokens_before": 0, "tokens_after": 0}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate == 0.6
    assert "压缩前 100 -> 压缩后 40" in row and "1 条计入" in row


def test_a_request_that_never_compressed_is_not_averaged_in_as_zero():
    """「这一次没压」不能当成降幅 0% 平均进去（#57）—— 那正是把「没发生」
    写成「发生了但结果是 0」。要出「不适用 + 为什么」，而且不许提分词器。"""
    rep = run_eval(GOLDEN, _ask({
        "tokens_before": None, "tokens_after": None, "tokenizer": "Qwen/x", "budget": 3000,
        "no_compress_note": "历史 2824 tokens 未超预算 3000，不需要压缩"}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate is None
    assert "不适用" in row and "未超预算" in row and "不需要压缩" in row
    assert "0%" not in row
    assert "没有真实分词器" not in row


def test_a_failed_summarizer_is_reported_as_a_failure_not_as_a_small_history():
    """真实原因必须**照抄**、不许反推：摘要炸了就得写「摘要失败」。

    从「没有摘要」反推「未超预算」，会印出「历史 N tokens 未超预算 M」而 N > M。
    """
    rep = run_eval(GOLDEN, _ask({
        "tokens_before": None, "tokens_after": None, "tokenizer": "Qwen/x", "budget": 3,
        "no_compress_note": "摘要失败，退回不压缩：RuntimeError: 网络挂了"}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert "不适用" in row and "摘要失败" in row and "RuntimeError" in row
    assert "未超预算" not in row


def test_only_the_items_that_really_compressed_count_toward_the_reduction():
    """一部分触发、一部分没触发：只有触发的进降幅。

    把没触发的也拉进来（它 before == after）会把降幅稀释成一半 ——
    那不是「压缩省得少」，是分母里混了根本没压的那几条。
    """
    rep = run_eval([GOLDEN[0], {**GOLDEN[0], "question": "b"}], mix(
        {"tokens_before": 100, "tokens_after": 40},
        {"tokens_before": None, "tokens_after": None,
         "no_compress_note": "历史 10 tokens 未超预算 3000，压缩未触发"}))
    row = next(ln for ln in _lines(rep) if ln.startswith("上下文压缩降幅(token)"))

    assert rep.token_reduction_rate == 0.6          # 不是 0.3
    assert "1 条计入" in row
