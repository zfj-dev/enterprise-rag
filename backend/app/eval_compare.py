"""同一黄金集跑多条链路，把数字并排对比（票 15）。

只做**计时 + 排版**：每条链路都以 AnswerFn 注入，指标一律取自评测核心（票 01）——
对比表里另算一套公式，就等于有了两份口径。
率型指标写成百分点差（pp），三行延迟都写毫秒差；没分母的写 -，绝不写 0。
"""
from __future__ import annotations

import time

from typing import Callable, Sequence

from app.eval_core import AnswerFn, Report, percentile, run_eval


def _ratio(rate: float | None, items: Sequence) -> float | None:
    """有分母才给数字：0 分与「没测」不能长一个样。"""
    return None if rate is None or not items else float(rate)


def _rate_rows() -> tuple[tuple[str, Callable[[Report], float | None]], ...]:
    return (
        ("事实命中率", lambda r: _ratio(r.fact_rate, r.positives)),
        ("引用忠实度", lambda r: _ratio(r.grounded_rate, r.positives)),
        ("引用页码正确率", lambda r: _ratio(r.page_rate, [x for x in r.positives if x.has_page])),
        ("拒答率(负样本)", lambda r: _ratio(r.refuse_rate, r.negatives)),
        ("引用覆盖率(均值)", lambda r: _ratio(r.citation_coverage_rate, r.positives)),
    )


def _pct(rate: float | None) -> str:
    return "-" if rate is None else "%.0f%%" % (rate * 100)


def _ms(value: float | None) -> str:
    return "-" if value is None else "%.0f ms" % value


def _delta_pp(a: float | None, b: float | None) -> str:
    return "-" if a is None or b is None else "%+dpp" % round((b - a) * 100)


def _delta_ms(a: float | None, b: float | None) -> str:
    return "-" if a is None or b is None else "%+.0f ms" % (b - a)


def _timed(ask: AnswerFn) -> tuple[AnswerFn, list]:
    """跑一遍并逐条记挂钟 —— 每条链路都用同一把尺子量，才谈得上比。"""
    times: list[float] = []

    def wrapped(question: str) -> dict:
        began = time.perf_counter()
        try:
            return ask(question)
        finally:
            times.append((time.perf_counter() - began) * 1000)

    return wrapped, times


def compare_links(goldenset: Sequence[dict], links: dict[str, AnswerFn],
                  note: str | None = None) -> list[str]:
    """`{链路名: answer_fn}` -> 对比报告正文。字典顺序即报告里的列顺序。

    `note` 由调用方补一行口径（例如「这一列只含代理循环本身」）—— 口径跟数字写在一起才可信。
    """
    names = list(links)
    if len(names) < 2:
        raise ValueError("对比至少要两条链路，当前只有 %d 条" % len(names))
    reports: dict[str, Report] = {}
    spans: dict[str, list] = {}
    for name in names:
        ask, seen = _timed(links[name])
        reports[name] = run_eval(goldenset, ask)
        spans[name] = seen

    head = reports[names[0]]
    lines = [
        "=== 双链路对比（同一黄金集、同一个评测核心）===",
        "黄金集 %d 条：正样本 %d / 负样本 %d"
        % (head.total, len(head.positives), len(head.negatives)),
        "两条链路都按「问题 -> 答案 + 来源」注入核心，评测代码零改动 —— 数字同口径可比",
        "变化列 = %s -> %s" % (names[0], names[-1]),
        "",
        ("%-18s" % "指标") + "".join("%16s" % n for n in names) + "%12s" % "变化",
    ]
    for label, get in _rate_rows():
        cells = [get(reports[n]) for n in names]
        lines.append(("%-18s" % label) + "".join("%16s" % _pct(v) for v in cells)
                     + "%12s" % _delta_pp(cells[0], cells[-1]))

    stats = {n: (sum(spans[n]) / len(spans[n]) if spans[n] else None,
                 percentile(spans[n], 50), percentile(spans[n], 95)) for n in names}
    for row, idx in (("平均总耗时", 0), ("总耗时 P50", 1), ("总耗时 P95", 2)):
        cells = [stats[n][idx] for n in names]
        lines.append(("%-18s" % row) + "".join("%16s" % _ms(v) for v in cells)
                     + "%12s" % _delta_ms(cells[0], cells[-1]))
    lines.append("")
    lines.append("口径：延迟是**单次顺序**跑的挂钟（每条一次），不是并发下的分位；"
                 "正确性判据全部来自评测核心")
    if note:
        lines.append("      " + note)

    for name in names:
        lines += ["", "=== %s · 明细 ===" % name]
        lines.extend(reports[name].to_lines())
    return lines
