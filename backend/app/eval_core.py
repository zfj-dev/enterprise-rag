"""评测核心（票 01）：注入式纯函数 —— 给任意「问答实现」算出一份确定的评测数字。

`run_eval(黄金集, answer_fn, judge_fn)` 的**被评对象与裁判都从外部注入**，所以核心
**不连服务、不调 LLM**：离线、CI、测试里都能跑出确定结果，换个后端也不用改评测代码。

判据口径（免 LLM）：把期望事实与待查文本都「去空白 + 转小写」后做子串匹配 ——
解析器会在数字/标点之间插空格（'表 3 . 1'、'Windows 11'），不这样归一化会整片漏判。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

# 问题 -> {"answer": str, "sources": [{"text": str, "page": int, ...}]}
AnswerFn = Callable[[str], dict]
# (问题, 答案, 来源) -> 裁判结论（RAGAS 之类；其内容由票 05 定义）
JudgeFn = Callable[[str, str, list], dict]


def normalize(text) -> str:
    """判据口径：去空白 + 转小写。"""
    return "".join(str(text or "").split()).lower()


@dataclass
class ItemResult:
    question: str
    expect: str
    answer: str
    fact_hit: bool              # 期望事实出现在答案里
    grounded: bool              # 期望事实出现在引用来源文本里（引用忠实度）
    expect_page: int | None
    pages: list                 # 来源里的页码
    page_hit: bool | None       # 期望页码是否在来源页码里；本条没声明页码时为 None
    judged: dict | None = None  # 注入 judge_fn 时的裁判结论


@dataclass
class Report:
    items: list[ItemResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    def _rate(self, picked: Callable[[ItemResult], bool]) -> float:
        return sum(1 for x in self.items if picked(x)) / max(1, len(self.items))

    @property
    def fact_rate(self) -> float:
        return self._rate(lambda x: x.fact_hit)

    @property
    def grounded_rate(self) -> float:
        return self._rate(lambda x: x.grounded)

    @property
    def page_rate(self) -> float | None:
        scored = [x for x in self.items if x.page_hit is not None]
        if not scored:
            return None
        return sum(1 for x in scored if x.page_hit) / len(scored)

    def to_lines(self) -> list[str]:
        """报告正文：先口径、再逐条、后汇总 —— 数字脱离口径就不可信。"""
        lines = [
            "判据口径：期望事实与待查文本都「去空白 + 转小写」后做子串匹配",
            "  fact_hit  期望事实出现在答案里",
            "  grounded  期望事实出现在引用来源文本里（引用忠实度）",
            "  page_hit  期望页码出现在引用来源页码里",
            "",
        ]
        for x in self.items:
            tail = (" | 页码:%s->%s" % (x.expect_page, "✓" if x.page_hit else x.pages)
                    if x.expect_page else " | 来源页:%s" % (x.pages,))
            lines.append("[%s] Q:%s | 期望:%s | 命中:%s|忠实:%s%s"
                         % ("PASS" if x.fact_hit else "FAIL",
                            x.question, x.expect, x.fact_hit, x.grounded, tail))
            lines.append("    答案前90字: %s" % x.answer[:90].replace(chr(10), " / "))
        lines.append("")
        lines.append("结果: 答案含期望事实 %d%%  (%d/%d)"
                     % (round(self.fact_rate * 100),
                        sum(1 for x in self.items if x.fact_hit), self.total))
        lines.append("引用忠实度(事实在来源里) %d%%  (%d/%d)"
                     % (round(self.grounded_rate * 100),
                        sum(1 for x in self.items if x.grounded), self.total))
        if self.page_rate is not None:
            scored = [x for x in self.items if x.page_hit is not None]
            lines.append("引用页码正确 %d%%  (%d/%d)"
                         % (round(self.page_rate * 100),
                            sum(1 for x in scored if x.page_hit), len(scored)))
        return lines


def run_eval(goldenset: Sequence[dict], answer_fn: AnswerFn,
             judge_fn: JudgeFn | None = None) -> Report:
    """对黄金集逐条跑 answer_fn，按固定口径判据算出报告。"""
    report = Report()
    for g in goldenset:
        question = g.get("question", "")
        expect = g.get("expect", "")
        expect_page = g.get("page")
        out = answer_fn(question) or {}
        answer = out.get("answer", "") or ""
        sources = [s for s in (out.get("sources") or []) if isinstance(s, dict)]

        want = normalize(expect)
        src_text = " ".join(str(s.get("text", "")) for s in sources)
        pages = [s.get("page") for s in sources]

        report.items.append(ItemResult(
            question=question, expect=expect, answer=answer,
            # 期望事实缺失 → 记为未命中，而不是静默跳过（否则分母变小、数字虚高）
            fact_hit=bool(want) and want in normalize(answer),
            grounded=bool(want) and want in normalize(src_text),
            expect_page=expect_page, pages=pages,
            page_hit=(expect_page in pages) if expect_page else None,
        ))
        if judge_fn is not None:
            report.items[-1].judged = judge_fn(question, answer, sources)
    return report
