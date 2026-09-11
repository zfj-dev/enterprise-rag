"""双链路对比报告（票 15 / #22）：同一黄金集、同一个评测核心，两条链路的数字并排看。"""
from __future__ import annotations

from app.eval_compare import compare_links

GOLDEN = [
    {"question": "营收多少", "expect": "803.96", "page": 1},
    {"question": "利润多少", "expect": "100", "page": 2},
    {"question": "有没有提到火星基地", "negative": True},
]


def _hit(question):
    return {"answer": "营收803.96亿元。", "sources": [{"text": "营收803.96亿元", "page": 1}],
            "citation_coverage": 0.5}


def _miss(question):
    return {"answer": "无法确定（未检索到可引用的内容）。", "sources": []}


def test_both_links_are_reported_side_by_side():
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss})
    text = chr(10).join(lines)

    assert "确定性链路" in text and "代理链路" in text
    assert "事实命中率" in text


def test_the_numbers_are_the_eval_core_ones():
    """对比表里的数字必须来自 core —— 另算一套公式就等于有两份口径。"""
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss})
    text = chr(10).join(lines)

    assert "50%" in text          # 确定性链路：2 条正样本命中 1 条
    assert "0%" in text           # 代理链路：一条都没命中
    assert "100%" in text         # 两条链路都拒答了负样本


def test_negative_samples_go_to_refusal_not_fact_rate():
    lines = compare_links(GOLDEN, {"确定性链路": _miss, "代理链路": _hit})
    text = chr(10).join(lines)

    assert "拒答率" in text
    assert "2 条正样本 / 1 条负样本" in text or "正样本 2 / 负样本 1" in text


def test_delta_column_is_in_percentage_points():
    """变化列是百分点差 —— 出现裸小数（0.5）就等于两边的读者各理解一套口径。"""
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss})
    text = chr(10).join(lines)

    assert "-50pp" in text        # 代理 0% - 确定性 50%


def test_latency_is_reported_per_link():
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss})
    text = chr(10).join(lines)

    assert "平均总耗时" in text and "ms" in text
    assert "P50" in text and "P95" in text


def test_missing_denominators_are_dashed_not_zero():
    """没分母的指标写 -，不写 0 —— 0 分和「没测」是两回事。"""
    lines = compare_links([{"question": "Q", "expect": "甲"}],
                          {"确定性链路": _hit, "代理链路": _miss})
    text = chr(10).join(lines)

    assert "拒答率(负样本)" in text
    assert "-" in text


def test_a_single_link_is_rejected():
    """一条链路谈不上对比 —— 直接抛，而不是出一张只有一列的表。"""
    import pytest

    with pytest.raises(ValueError):
        compare_links(GOLDEN, {"只有一条链路": _hit})


def test_all_latency_rows_carry_a_delta():
    """ROADMAP 要的是「延迟变化」—— 均值 / P50 / P95 三行都得有变化列。"""
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss})
    rows = [ln for ln in lines if ln.lstrip().startswith(("平均总耗时", "总耗时"))]

    assert len(rows) == 3
    assert all(ln.rstrip().endswith("ms") for ln in rows)


def test_caller_note_sits_next_to_the_numbers():
    """口径跟数字写在一起才可信 —— 调用方补的那行要真的出现。"""
    lines = compare_links(GOLDEN, {"确定性链路": _hit, "代理链路": _miss},
                          note="代理列只计代理循环本身")

    assert any("代理列只计代理循环本身" in ln for ln in lines)
