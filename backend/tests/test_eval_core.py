"""评测核心（票 01 / #8）：注入式纯函数 —— 不连服务、不调 LLM。

被评的问答函数与裁判函数都注入 stub，因此断言是确定的、离线的。
"""
from __future__ import annotations

import pytest

from app.eval_core import normalize, run_eval

GOLDEN = [
    {"question": "Q1", "expect": "803.96"},
    {"question": "Q2", "expect": "PyCharm", "page": 29},
    {"question": "Q3", "expect": "RTX 3060", "page": 31},
]

ANSWERS = {
    "Q1": {"answer": "803.96 亿元 [来源: a.pdf, 第1页]",
           "sources": [{"text": "营业收入 803.96 亿元", "page": 1}]},
    "Q2": {"answer": "用的 PyCharm [来源: b.pdf, 第29页]",
           "sources": [{"text": "PyCharm", "page": 29}]},
    "Q3": {"answer": "不知道", "sources": [{"text": "与本问无关", "page": 1}]},
}


def _answer_fn(table):
    return lambda question: table[question]


# ---------- 指标计算 ----------

def test_fact_grounded_and_page_rates():
    rep = run_eval(GOLDEN, _answer_fn(ANSWERS))

    assert rep.total == 3
    assert [x.fact_hit for x in rep.items] == [True, True, False]      # Q3 答案里没有期望事实
    assert [x.grounded for x in rep.items] == [True, True, False]      # 且来源里也没有
    assert rep.fact_rate == pytest.approx(2 / 3)
    assert rep.grounded_rate == pytest.approx(2 / 3)
    assert rep.page_rate == pytest.approx(1 / 2)                       # 只有 Q2 页码对得上


def test_page_rate_is_none_when_no_item_declares_a_page():
    rep = run_eval([{"question": "Q1", "expect": "803.96"}], _answer_fn(ANSWERS))
    assert rep.page_rate is None
    assert rep.items[0].page_hit is None

    text = "\n".join(rep.to_lines())
    assert "引用页码正确 不适用" in text and "分母为 0" in text


def test_empty_goldenset_is_all_zero():
    rep = run_eval([], lambda q: {"answer": "", "sources": []})
    assert rep.total == 0
    assert rep.fact_rate == 0.0 and rep.grounded_rate == 0.0 and rep.page_rate is None


def test_missing_expect_is_a_miss_not_a_skip():
    """期望事实缺失时标为未命中，而不是静默跳过（否则分母会变小、数字虚高）。"""
    golden = [{"question": "Q", "expect": ""}, {"question": "R", "expect": "甲"}]
    table = {"Q": {"answer": "随便", "sources": []}, "R": {"answer": "甲", "sources": []}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.total == 2
    assert rep.items[0].fact_hit is False
    assert rep.fact_rate == pytest.approx(0.5)
    assert rep.missing_expect_count == 1

    text = "\n".join(rep.to_lines())
    assert "(未写期望事实)" in text and "按未命中计" in text


def test_answer_fn_may_omit_sources():
    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"})
    assert rep.items[0].fact_hit is True
    assert rep.items[0].grounded is False
    assert rep.items[0].pages == []


# ---------- 判据口径 ----------

def test_normalize_removes_whitespace_and_case():
    assert normalize("表 4 . 1") == normalize("表4.1")
    assert normalize("Windows  11") == normalize("windows11")
    assert normalize(None) == ""
    assert normalize("RTX 3060") == "rtx3060"


def test_parser_inserted_spaces_do_not_break_the_hit():
    """解析器会在数字/标点间插空格（'表 3 . 1'），判据必须照命中。"""
    golden = [{"question": "表3.1 的实验环境", "expect": "RTX 3060", "page": 29}]
    table = {"表3.1 的实验环境": {"answer": "实验用 RTX  3060 显卡",
                                  "sources": [{"text": "GPU：RTX 3060", "page": 29}]}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.items[0].fact_hit is True
    assert rep.items[0].grounded is True
    assert rep.items[0].page_hit is True


# ---------- 裁判缝 ----------

def test_judge_fn_is_called_per_item_and_recorded():
    seen = []

    def judge(question, answer, sources):
        seen.append(question)
        return {"faithful": True}

    rep = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=judge)

    assert seen == ["Q1", "Q2", "Q3"]
    assert [x.judged for x in rep.items] == [{"faithful": True}] * 3


def test_judge_receives_the_answer_and_sources():
    got = {}

    def judge(question, answer, sources):
        got["question"], got["answer"], got["sources"] = question, answer, sources
        return {}

    run_eval([GOLDEN[0]], _answer_fn(ANSWERS), judge_fn=judge)
    assert got["question"] == "Q1"
    assert got["answer"] == ANSWERS["Q1"]["answer"]
    assert got["sources"] == ANSWERS["Q1"]["sources"]


def test_no_judge_fn_leaves_judged_none():
    rep = run_eval(GOLDEN, _answer_fn(ANSWERS))
    assert all(x.judged is None for x in rep.items)


def test_stub_run_is_deterministic():
    """stub 注入下同一输入两次跑出完全一样的报告（无网络、无真实 LLM）。"""
    a = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=lambda q, a_, s: {"n": 1})
    b = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=lambda q, a_, s: {"n": 1})
    assert a.to_lines() == b.to_lines()


# ---------- 报告文本 ----------

def test_report_lines_state_the_criterion_and_the_numbers():
    text = "\n".join(run_eval(GOLDEN, _answer_fn(ANSWERS)).to_lines())

    assert "口径" in text and "去空白" in text          # 数字要有口径，才可信
    assert "67%" in text                                # 答案含期望事实 2/3
    assert "50%" in text                                # 引用页码正确 1/2
    assert "Q3" in text                                 # 逐条都在


# ---------- 缺失时的口径：不静默跳过 ----------

def test_item_without_declared_page_is_stated_not_silently_dropped():
    """没声明页码的条目不计入页码率 —— 但报告必须写明条数，不能看着像被跳过。"""
    golden = [{"question": "Q1", "expect": "甲"}, {"question": "Q2", "expect": "乙", "page": 5}]
    table = {"Q1": {"answer": "甲", "sources": [{"text": "甲", "page": 1}]},
             "Q2": {"answer": "乙", "sources": [{"text": "乙", "page": 5}]}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.total == 2
    assert rep.undeclared_page_count == 1
    assert rep.page_rate == 1.0                       # 分母只算声明了页码的那条
    text = "\n".join(rep.to_lines())
    assert "另有 1 条黄金集条目未声明页码" in text
    assert "未声明页码" in text                        # 逐条那行也点名了
def test_item_whose_run_produced_nothing_is_a_miss_not_a_skip():
    """答案空、来源空：声明了期望与页码的那条要判成未命中，而不是消失。"""
    rep = run_eval([{"question": "Q", "expect": "甲", "page": 3}],
                   lambda q: {"answer": "", "sources": []})

    assert rep.total == 1
    assert rep.items[0].fact_hit is False
    assert rep.items[0].grounded is False
    assert rep.items[0].page_hit is False             # 声明了页码却没有来源页码
    assert rep.page_rate == 0.0


def test_all_three_numbers_always_get_a_line():
    """三个数字必须行行都在 —— 空黄金集也不例外。"""
    text = "\n".join(run_eval([], lambda q: {}).to_lines())
    assert "答案含期望事实" in text
    assert "引用忠实度" in text
    assert "引用页码正确" in text
