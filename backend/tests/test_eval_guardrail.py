"""质量护栏（票 22 / #29）：同一黄金集跑**压缩前 / 压缩后** —— 事实命中不下降才算通过。

护栏结论是**纯函数**（app/eval_compare.guardrail_lines）：输入两份报告，输出结论。
降幅必须与事实命中**并列** —— 只报降幅不报质量，等于奖励「把上下文砍掉」。
"""
from __future__ import annotations

import json

from app.eval_compare import guardrail_lines, render_compare, run_links
from app.eval_core import ItemResult, Report


def _item(question: str, hit: bool, *, ctx=None) -> ItemResult:
    """一条正样本的评测结果；ctx=(压缩前, 压缩后, 分词器) 用来验降幅并列。"""
    before, after, tok = ctx if ctx else (None, None, "")
    return ItemResult(question=question, expect="803.96",
                      answer="营收803.96亿元" if hit else "不清楚",
                      fact_hit=hit, grounded=hit, expect_page=None, pages=[1],
                      page_hit=None, ctx_before=before, ctx_after=after, ctx_tokenizer=tok)


def _report(hits, *, ctx=None) -> Report:
    return Report(items=[_item("Q%d" % i, h, ctx=ctx) for i, h in enumerate(hits)])


def _text(lines) -> str:
    return "\n".join(lines)


# ---------- 结论：事实命中不下降才通过 ----------

def test_a_drop_in_fact_hit_fails_the_guardrail():
    """压缩后事实命中掉了 —— 明确标未通过（降幅再漂亮也不算数）。"""
    text = _text(guardrail_lines(_report([True, True, True, True]),
                                 _report([True, True, False, False])))

    assert "未通过" in text
    assert "下降" in text and "50pp" in text          # 100% -> 50%


def test_an_unchanged_fact_hit_passes():
    text = _text(guardrail_lines(_report([True, False]), _report([True, False])))

    assert "通过" in text and "未通过" not in text


def test_an_improvement_passes_and_says_so():
    text = _text(guardrail_lines(_report([False, False]), _report([True, True])))

    assert "通过" in text and "上升" in text


def test_a_pass_with_zero_on_both_sides_is_flagged_as_uninformative():
    """两边都是 0% 也算「未下降」，但这等于什么都没证明 —— 必须说出来，别当成绩。"""
    text = _text(guardrail_lines(_report([False, False]), _report([False, False])))

    assert "通过" in text and "没有信息量" in text


def test_without_positives_the_guardrail_is_undecidable_not_passing():
    """没有正样本就没有分母 —— 不许写成「通过」。"""
    text = _text(guardrail_lines(Report(items=[]), Report(items=[])))

    assert "不可判定" in text and "通过" not in text


def test_a_pass_that_compressed_nothing_is_flagged():
    """有分词器却一条都没压到（降幅不适用）—— 这样的「通过」证明不了压缩安全。"""
    text = _text(guardrail_lines(_report([True, True], ctx=(0, 0, "Qwen/x")),
                                 _report([True, True], ctx=(0, 0, "Qwen/x"))))

    assert "通过" in text
    assert "没有真的压到东西" in text


def test_a_swap_is_disclosed_even_when_the_total_is_flat():
    """一条升一条降、合计持平 —— 不许当成「没变」（只看合计会把置换当没事）。"""
    text = _text(guardrail_lines(_report([True, False]), _report([False, True])))

    assert "通过" in text
    assert "1 条由未命中变命中" in text and "1 条由命中变未命中" in text


# ---------- 降幅与事实命中并列 ----------

def test_the_reduction_is_reported_next_to_the_fact_hit():
    """降幅取自**压缩后**那一侧，且与事实命中写在同一段里。"""
    lines = guardrail_lines(_report([True], ctx=(100, 100, "Qwen/x")),
                            _report([True], ctx=(100, 40, "Qwen/x")))
    text = _text(lines)

    assert "事实命中" in text
    assert "60%" in text and "Qwen/x" in text          # 1 - 40/100，口径跟着数字走


def test_the_verdict_stands_on_quality_even_when_there_is_no_reduction_number():
    """没接真实分词器时降幅写「不可用」，但结论照样得给出 —— 不许因为没降幅就不判。"""
    text = _text(guardrail_lines(_report([True, True]), _report([True, True])))

    assert "事实命中" in text and "通过" in text
    assert "不可用" in text


def test_a_failure_does_not_dress_up_the_reduction_as_a_result():
    """未通过时，降幅必须被写成「不算数」，不能与结论并列成两个平级成果。"""
    text = _text(guardrail_lines(_report([True, True], ctx=(100, 100, "Qwen/x")),
                                 _report([True, False], ctx=(100, 40, "Qwen/x"))))

    assert "未通过" in text
    assert "不算数" in text and "60%" in text


# ---------- 跑得出报告，且不用跑第二遍 ----------

def _ask(answer: str):
    def ask(question: str) -> dict:
        return {"answer": answer, "sources": [{"text": answer, "page": 1}]}
    return ask


def test_run_links_hands_back_the_reports_the_guardrail_needs():
    """对比正文与护栏结论共用同一次跑 —— 报告要能直接喂给 guardrail_lines。"""
    golden = [{"question": "Q", "expect": "甲"}]
    reports, spans = run_links(golden, {"压缩前": _ask("甲"), "压缩后": _ask("甲")})

    assert set(reports) == {"压缩前", "压缩后"}
    assert reports["压缩前"].fact_rate == 1.0
    assert len(spans["压缩后"]) == 1                  # 每条链路各记一次挂钟
    assert "事实命中率" in _text(render_compare(reports, spans))
    assert "通过" in _text(guardrail_lines(reports["压缩前"], reports["压缩后"]))


def test_a_single_link_is_rejected_before_anything_runs():
    """一条链路谈不上对比 —— 跑之前就拦下，别白跑一遍黄金集再报错。"""
    import pytest

    calls: list = []

    def ask(question: str) -> dict:
        calls.append(question)
        return {"answer": "甲", "sources": []}

    with pytest.raises(ValueError):
        run_links([{"question": "Q", "expect": "甲"}], {"只有一条链路": ask})

    assert calls == []


# ---------- runner：两条配置真的各跑一遍 ----------

DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。"


def _inputs(tmp_path, expect: str = "模拟回答"):
    doc = tmp_path / "annual.txt"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "营收多少？", "expect": expect, "page": 1}],
                                 ensure_ascii=False), encoding="utf-8")
    return str(golden), str(doc), str(tmp_path / "guardrail.log")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_missing_inputs_are_reported_as_not_run(tmp_path):
    import evaluate_guardrail

    _, _, report = _inputs(tmp_path)
    evaluate_guardrail.main(golden=str(tmp_path / "nope.json"), doc=str(tmp_path / "nope.pdf"),
                            report=report)
    text = _read(report)

    assert "未跑" in text
    assert "事实命中" not in text and "%" not in text


def test_both_configurations_run_offline_and_the_guardrail_concludes(tmp_path):
    """压缩前 / 压缩后各跑一遍，报告里给出结论与跑法（单轮没有可压的历史）。"""
    import evaluate_guardrail

    golden, doc, report = _inputs(tmp_path)
    evaluate_guardrail.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "压缩前" in text and "压缩后" in text
    assert "事实命中" in text and "通过" in text
    assert "多轮" in text
    assert "演示模式" in text                          # 假模型的数字不许被当真


def test_the_compress_switch_is_restored_after_the_run(tmp_path):
    """跑完必须把全局压缩开关还原 —— 评测脚本不许改坏接下来的运行。"""
    import evaluate_guardrail
    from app.config import get_settings

    golden, doc, report = _inputs(tmp_path)
    before = get_settings().context_compress
    try:
        evaluate_guardrail.main(golden=golden, doc=doc, report=report)
        assert get_settings().context_compress is before
    finally:
        get_settings().context_compress = before
