"""同一黄金集跑多条链路，把数字并排对比（票 15）；外加压缩的**质量护栏**结论（票 22）。

只做**计时 + 排版**：每条链路都以 AnswerFn 注入，指标一律取自评测核心（票 01）——
对比表里另算一套公式，就等于有了两份口径。
率型指标写成百分点差（pp），三行延迟都写毫秒差；没分母的写 -，绝不写 0。

`run_links` 跑、`render_compare` 排、`compare_links` 是两者的合成；质量护栏（`guardrail_lines`）
拿前两者的报告判「压缩后事实命中有没有下降」，因此不必把黄金集再跑一遍。
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


def _require_two(names: Sequence) -> None:
    """一条链路谈不上对比 —— 跑之前就拦下，别白跑一遍黄金集再报错。"""
    if len(names) < 2:
        raise ValueError("对比至少要两条链路，当前只有 %d 条" % len(names))


def run_links(goldenset: Sequence[dict], links: dict[str, AnswerFn]
              ) -> tuple[dict[str, Report], dict[str, list]]:
    """跑每条链路，返回 (报告, 每次提问的挂钟)。

    指标一律来自评测核心；报告与挂钟分开交回 —— 对比正文之外还要拿报告做判据的场景
    （质量护栏看的是事实命中）不必把黄金集跑第二遍。
    """
    _require_two(links)
    reports: dict[str, Report] = {}
    spans: dict[str, list] = {}
    for name, ask in links.items():
        timed, seen = _timed(ask)
        reports[name] = run_eval(goldenset, timed)
        spans[name] = seen
    return reports, spans


def render_compare(reports: dict[str, Report], spans: dict[str, list],
                   note: str | None = None) -> list[str]:
    """把**已经跑出来的**报告排成对比正文。字典顺序即报告里的列顺序。

    `note` 由调用方补一行口径（例如「这一列只含代理循环本身」）—— 口径跟数字写在一起才可信。
    """
    _require_two(reports)
    names = list(reports)

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


def compare_links(goldenset: Sequence[dict], links: dict[str, AnswerFn],
                  note: str | None = None) -> list[str]:
    """`{链路名: answer_fn}` -> 对比报告正文（跑一遍 + 排版）。"""
    reports, spans = run_links(goldenset, links)
    return render_compare(reports, spans, note=note)


# ---------- 质量护栏（票 22；判据与记忆护栏共用）----------

def quality_verdict(name: str, first: float, second: float, *, failure_note: str = "") -> str:
    """同一个质量指标「改前 -> 改后」的通用结论（压缩护栏与记忆护栏共用这一处）。

    下降就是**未通过**；没有分母的调用方自己先判「不可判定」，别拿这里当兜底。
    两边都是 0 时如实标注「没有信息量」—— 0 和 0 相等不构成任何证据。
    """
    if first == 0 and second == 0:
        return "通过 —— 但两边都是 0%，这条通过没有信息量，先确认黄金集与模型可用"
    delta = second - first
    if delta < -1e-9:
        return "**未通过** —— %s下降 %dpp%s" % (name, round(-delta * 100), failure_note)
    if delta > 1e-9:
        return "通过 —— %s还上升了 %dpp" % (name, round(delta * 100))
    return "通过 —— %s未下降" % name


def _reduction_phrase(report: Report) -> str:
    """降幅那一句：有数字就带上口径，没有就照评测核心给的原因如实写。"""
    rate = report.token_reduction_rate
    if rate is None:
        return report.reduction_missing_reason
    return "%.0f%%  (口径 %s；%d 条计入)" % (round(rate * 100), report.tokenizer_label,
                                          report.token_reduction_count)


def _fact_moves(before: Report, after: Report) -> tuple[list[str], list[str]]:
    """逐条看事实命中怎么动的（两份报告按黄金集顺序对齐）。

    整体持平也可能是「一条升、一条降」—— 只看合计会把这种置换当成没变。
    """
    lost, gained = [], []
    for b, a in zip(before.positives, after.positives):
        if b.fact_hit and not a.fact_hit:
            lost.append(a.question)
        elif a.fact_hit and not b.fact_hit:
            gained.append(a.question)
    return lost, gained


def guardrail_lines(before: Report, after: Report) -> list[str]:
    """质量护栏结论（票 22）：**事实命中不下降**才算通过，降幅与它**并列**呈现。

    「压缩」这类优化最容易把降 token 当成成果，所以这里的判据是**质量**：事实命中掉了
    就明确写「未通过」，并说明降幅不算数 —— 只报降幅不报质量，等于奖励「把上下文砍掉」。
    没有分母（黄金集里没有正样本）时写「不可判定」，绝不写「通过」；这轮压根没压到东西
    时（降幅不适用）也要说清 —— 那样的「通过」证明不了压缩安全。
    """
    first = _ratio(before.fact_rate, before.positives)
    second = _ratio(after.fact_rate, after.positives)
    lines = ["", "=== 质量护栏（压缩前 -> 压缩后）==="]
    if first is None or second is None:
        lines.append("  不可判定 —— 黄金集里没有正样本，事实命中率没有分母；降幅不构成结论")
        lines.append("  压缩降幅 %s" % _reduction_phrase(after))
        return lines

    lines.append("  事实命中 压缩前 %.0f%% -> 压缩后 %.0f%%   %s"
                 % (first * 100, second * 100,
                    quality_verdict("事实命中", first, second, failure_note="；压缩降幅不算数")))
    lines.append("  压缩降幅 %s" % _reduction_phrase(after))

    lost, gained = _fact_moves(before, after)
    if lost and gained:
        lines.append("  逐条变化： %d 条由未命中变命中、%d 条由命中变未命中 —— 整体持平不等于没变，"
                     "看下面的明细" % (len(gained), len(lost)))
    elif lost:
        lines.append("  逐条变化： %d 条由命中变未命中" % len(lost))
    elif gained:
        lines.append("  逐条变化： %d 条由未命中变命中" % len(gained))

    if after.tokenizer_label and after.token_reduction_rate is None:
        lines.append("  注意：这一轮**没有真的压到东西**（降幅不适用）——"
                     "这条通过只能证明链路跑得通，证明不了压缩安全")
    lines.append("  判据：两侧都只用评测核心的数字；降幅与事实命中**一起看** —— 单看降幅不算成果")
    return lines


def memory_guardrail_lines(before: Report, after: Report) -> list[str]:
    """记忆护栏（票 26）：**答案侧**的引用覆盖率不下降才算通过 —— 防「记得更多 = 编得更多」。

    为什么不判来源侧的「引用忠实度」：记忆**不进入 `sources`** 是构造性的硬约束，
    来源侧那些指标（grounded / page）**不可能**因记忆变化 —— 判它们等于自证，测不出任何东西。
    真正会被记忆推动的是**答案**：把记忆当依据写进去，论断就失去来源支撑，覆盖率随之下降。
    覆盖率拿不到（演示模式 / 免 LLM 跑法）时写「不可判定」，绝不写「通过」。
    """
    lines = ["", "=== 记忆护栏（记忆关闭 -> 记忆开启）==="]

    cov_b, cov_a = before.citation_coverage_rate, after.citation_coverage_rate
    if cov_b is None or cov_a is None:
        lines.append("  引用覆盖率(答案侧) %s -> %s   不可判定 —— 这两次都没拿到答案侧的覆盖率"
                     "（演示模式 / 免 LLM 跑法）；「记得更多 = 编得更多」这条要在真实模型下才验得了"
                     % (_pct(cov_b), _pct(cov_a)))
    else:
        lines.append("  引用覆盖率(答案侧) 记忆关闭 %.0f%% -> 记忆开启 %.0f%%   %s"
                     % (cov_b * 100, cov_a * 100, quality_verdict("引用覆盖率", cov_b, cov_a)))

    judged = False
    for name, b, a in (("引用忠实度(来源侧)", before.grounded_rate, after.grounded_rate),
                       ("事实命中", before.fact_rate, after.fact_rate)):
        fb = _ratio(b, before.positives)
        fa = _ratio(a, after.positives)
        if fb is None or fa is None:
            continue
        judged = True
        lines.append("  %s 记忆关闭 %.0f%% -> 记忆开启 %.0f%%  %s"
                     % (name, fb * 100, fa * 100, quality_verdict(name, fb, fa)))
    if not judged:
        lines.append("  不可判定 —— 黄金集里没有正样本，上面两项都没有分母")
    lines.append("  说明：来源侧的引用忠实度**不可能**因记忆而变（记忆不进入 sources 是构造性的）"
                 "—— 它不下降是设计保证，不是测出来的；会动的是答案侧的覆盖率。")
    lines.append("  硬约束：记忆可以进 prompt，但**不进入 sources** —— 回答的依据仍必须来自文档")
    lines.append("  判据：两侧都只用评测核心的数字；记得多不等于答得好，覆盖率掉下来就不算数")
    return lines
