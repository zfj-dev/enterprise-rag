"""跨会话记忆的交付物（票 26 / #33）：可复现的召回示例 + 引用覆盖率护栏。

示例走**真链路**（抽取 → 落库 → 召回 → 注入 prompt），只把 LLM 与抽取器换成确定性替身 ——
离线、可复现。护栏复用票 22 那一套判据（`quality_verdict`），换的只是被判的指标。
"""
from __future__ import annotations

import json

from app.eval_compare import memory_guardrail_lines, quality_verdict
from app.eval_core import ItemResult, Report


def _item(question: str, *, fact=True, grounded=True, coverage=None) -> ItemResult:
    return ItemResult(question=question, expect="803.96", answer="甲" if fact else "乙",
                      fact_hit=fact, grounded=grounded, expect_page=None, pages=[1],
                      page_hit=None, citation_coverage=coverage)


def _report(items) -> Report:
    return Report(items=list(items))


def _text(lines) -> str:
    return "\n".join(lines)


# ---------- 判据是共用的那一处 ----------

def test_the_verdict_is_shared_with_the_compression_guardrail():
    """两个护栏判的是同一件事（质量不下降），判据只该有一份。"""
    assert "未通过" in quality_verdict("引用覆盖率", 1.0, 0.5)
    assert "未下降" in quality_verdict("引用覆盖率", 1.0, 1.0)
    assert "上升" in quality_verdict("引用覆盖率", 0.5, 1.0)


# ---------- 护栏：判的是答案侧，不是来源侧 ----------

def test_a_drop_in_answer_side_coverage_fails_the_memory_guardrail():
    """记忆让答案更「敢说」—— 论断失去来源支撑、覆盖率掉下来，就是未通过。"""
    before = _report([_item("Q", coverage=0.9), _item("P", coverage=0.9)])
    after = _report([_item("Q", coverage=0.4), _item("P", coverage=0.4)])
    text = _text(memory_guardrail_lines(before, after))

    assert "未通过" in text
    assert "引用覆盖率" in text and "50pp" in text


def test_the_source_side_metric_is_labelled_structurally_unchanged():
    """来源侧的引用忠实度**不可能**因记忆而动 —— 不许把它当测出来的通过。"""
    text = _text(memory_guardrail_lines(_report([_item("Q")]), _report([_item("Q")])))

    assert "引用忠实度(来源侧)" in text
    assert "构造性" in text and "不进入 sources" in text


def test_without_answer_side_coverage_it_is_undecidable_not_passing():
    """拿不到答案侧覆盖率（演示模式 / 免 LLM）→ 不可判定，不许拿来源侧那条凑一个「通过」。"""
    text = _text(memory_guardrail_lines(_report([_item("Q")]), _report([_item("Q")])))

    assert "不可判定" in text and "真实模型下才验得了" in text


def test_a_drop_in_fact_hit_also_fails():
    """记歪了会答错 —— 事实命中也得看。"""
    before = _report([_item("Q"), _item("P")])
    after = _report([_item("Q"), _item("P", fact=False)])
    text = _text(memory_guardrail_lines(before, after))

    assert "未通过" in text and "事实命中" in text


def test_an_unchanged_pair_passes():
    text = _text(memory_guardrail_lines(_report([_item("Q"), _item("P")]),
                                        _report([_item("Q"), _item("P")])))

    assert "通过" in text and "未通过" not in text


def test_without_positives_it_is_undecidable_not_passing():
    text = _text(memory_guardrail_lines(Report(items=[]), Report(items=[])))

    assert "不可判定" in text and "通过" not in text


def test_the_guardrail_states_that_memory_is_never_a_source():
    """硬约束写在结论旁边：记忆可以进 prompt，但**不许**进 sources。"""
    text = _text(memory_guardrail_lines(_report([_item("Q")]), _report([_item("Q")])))

    assert "sources" in text and "不进入" in text


# ---------- runner：示例 + 护栏都落盘 ----------

DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。"


def _inputs(tmp_path, expect: str = "模拟回答", question: str = "营收多少？"):
    doc = tmp_path / "annual.txt"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": question, "expect": expect, "page": 1}],
                                 ensure_ascii=False), encoding="utf-8")
    return str(golden), str(doc), str(tmp_path / "memory.log")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_missing_inputs_are_reported_as_not_run(tmp_path):
    import evaluate_memory

    _, _, report = _inputs(tmp_path)
    evaluate_memory.main(golden=str(tmp_path / "nope.json"), doc=str(tmp_path / "nope.pdf"),
                         report=report)
    text = _read(report)

    assert "未跑" in text
    assert "%" not in text


def test_the_cross_session_demo_reaches_the_prompt(tmp_path):
    """会话 A 告知 → 事实落库 → 会话 B（新会话）召回并进入模型看到的 prompt。"""
    import evaluate_memory

    golden, doc, report = _inputs(tmp_path)
    evaluate_memory.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "会话 A" in text and "会话 B" in text
    assert evaluate_memory.FACT in text                       # 落库的那条事实
    assert "模型看到的 prompt 里含 【已知信息】: 是" in text     # 真进了 prompt（不是打印的标签）
    assert evaluate_memory.MEMORY_ANSWER in text              # 替身模型据此作答


def test_the_report_computes_that_memory_stayed_out_of_sources(tmp_path):
    """硬约束要**实算**核验，不是照抄一句「不会」—— 两个用户都拿最新一条结果。"""
    import evaluate_memory

    golden, doc, report = _inputs(tmp_path)
    evaluate_memory.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "硬约束核验" in text and "记忆混进 sources 了吗 —— 没有" in text


def test_the_guardrail_runs_two_columns_and_says_what_it_can(tmp_path):
    import evaluate_memory

    golden, doc, report = _inputs(tmp_path)
    evaluate_memory.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "记忆关闭" in text and "记忆开启" in text
    assert "引用覆盖率(答案侧)" in text and "事实命中" in text
    assert "不可判定" in text          # 演示模式拿不到答案侧覆盖率 —— 如实说
    assert "演示模式" in text           # 假模型的数字不许被当真


def test_zero_injection_claims_are_established_not_asserted(tmp_path):
    """零注入时要说「两列逐项一致」，而且要真的比过 —— 结果写的是「是」。"""
    import evaluate_memory

    golden, doc, report = _inputs(tmp_path)
    evaluate_memory.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "本次记忆召回" in text
    assert "逐项" in text


def test_the_memory_store_is_cleaned_up_so_runs_repeat(tmp_path):
    """跑完把自己的记忆清掉 —— 不清的话每跑一次多堆一批，召回数与护栏都会漂。"""
    import evaluate_memory
    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.eval_setup import eval_user

    golden, doc, report = _inputs(tmp_path)
    evaluate_memory.main(golden=golden, doc=doc, report=report)

    db = SessionLocal()
    try:
        user = eval_user(db, evaluate_memory.USERNAME)
        assert build_runtime().memory_store.list(user.id) == []
    finally:
        db.close()


def test_the_memory_switch_is_restored_after_the_run(tmp_path):
    """跑完把全局记忆开关还原 —— 评测脚本不许改坏接下来的运行。"""
    import evaluate_memory
    from app.config import get_settings

    golden, doc, report = _inputs(tmp_path)
    was = get_settings().memory_enabled
    try:
        evaluate_memory.main(golden=golden, doc=doc, report=report)
        assert get_settings().memory_enabled is was
    finally:
        get_settings().memory_enabled = was
