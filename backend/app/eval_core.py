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


# 拒答措辞。只收明确表示「答不了」的说法，不收「不确定」「可能」这类正常答案里也有的词。
# 覆盖系统自己的话术（citation.apply_no_source_no_claim、prompt 规则 2/3）与云端模型常见的拒答说法。
_REFUSAL_MARKERS = (
    "无法确定", "无法回答", "无法提供", "无法给出", "无法从", "无法协助", "无法帮",
    "不能提供", "不便提供", "不予回答", "拒绝回答",
    "未检索到", "没有找到相关", "未找到相关", "未找到", "没有相关信息",
    "不足以为", "不足以回答", "不足以",
    "资料中没有", "资料中未", "文档中没有", "没有相关资料", "资料未提及",
    "未提及", "文中未", "未包含相关",
)

# 拒答是「一句话把人挡回去」，不会长篇大论。超过这个长度就不再当拒答看 ——
# 专门用来挡住「资料里没有…，不过据我所知…」这种**先拒后硬答**，那正是要抓的幻觉，
# 让它混进拒答会把拒答率抬成虚高，与这条指标的本意相反。
_REFUSAL_MAX_CHARS = 120


def is_refusal(answer: str) -> bool:
    """免 LLM 的拒答判据：答案**整段**就是一句拒答话术，才算「明确拒答」。

    口径写死、确定性可单测。空答案**不算**拒答 —— 那是没答，不是拒答。
    """
    a = normalize(answer)
    return bool(a) and len(a) <= _REFUSAL_MAX_CHARS and any(m in a for m in _REFUSAL_MARKERS)


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
    negative: bool = False      # 负样本：答案不在文档里，期望拒答（黄金集写 "negative": true）
    refused: bool = False       # 判据认定「明确拒答」
    judged: dict | None = None  # 注入 judge_fn 时的裁判结论

    @property
    def has_expect(self) -> bool:
        """黄金集条目写没写期望事实。"""
        return bool(normalize(self.expect))

    @property
    def has_page(self) -> bool:
        """黄金集条目声明没声明页码 —— 全模块只认这一处定义。"""
        return self.expect_page is not None



@dataclass
class Report:
    items: list[ItemResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def positives(self) -> list[ItemResult]:
        """正样本：要查事实的。负样本只判拒答，不进事实 / 页码的分母。"""
        return [x for x in self.items if not x.negative]

    @property
    def negatives(self) -> list[ItemResult]:
        """负样本：答案不在文档里，期望拒答。"""
        return [x for x in self.items if x.negative]

    def _rate(self, picked: Callable[[ItemResult], bool]) -> float:
        items = self.positives
        return sum(1 for x in items if picked(x)) / max(1, len(items))

    @property
    def fact_rate(self) -> float:
        return self._rate(lambda x: x.fact_hit)

    @property
    def grounded_rate(self) -> float:
        return self._rate(lambda x: x.grounded)

    @property
    def refused_count(self) -> int:
        """明确拒答的负样本条数 —— 拒答率与报告渲染共用这一处，免得两处各数一遍。"""
        return sum(1 for x in self.negatives if x.refused)

    @property
    def refuse_rate(self) -> float | None:
        """负样本里「明确拒答」的占比；没有负样本条目则为 None。"""
        negs = self.negatives
        if not negs:
            return None
        return self.refused_count / len(negs)

    @property
    def page_rate(self) -> float | None:
        scored = [x for x in self.positives if x.has_page]
        if not scored:
            return None
        return sum(1 for x in scored if x.page_hit) / len(scored)

    @property
    def missing_expect_count(self) -> int:
        """该写期望事实却没写的正样本 —— 按未命中计，但仍留在报告里（不静默跳过）。

        负样本本来就没有期望事实，不算数据缺口。
        """
        return sum(1 for x in self.positives if not x.has_expect)

    @property
    def undeclared_page_count(self) -> int:
        """没声明页码的正样本 —— 不计入页码率，但报告写明条数，看着不像被跳过。"""
        return sum(1 for x in self.positives if not x.has_page)

    def to_lines(self) -> list[str]:
        """报告正文：先口径、再逐条、后汇总 —— 数字脱离口径就不可信。"""
        lines = [
            "判据口径：期望事实与待查文本都「去空白 + 转小写」后做子串匹配",
            "  fact_hit  期望事实出现在答案里",
            "  grounded  期望事实出现在**随答案返回的来源文本**里（系统给出的来源集合；"
            "不逐条核对该论断是否被答案显式引用）",
            "  page_hit  期望页码出现在随答案返回的来源页码里（声明了页码却没来源页码 = 未命中）",
            "  黄金集条目缺期望事实 / 页码时：不跳过该条，而是按未命中计入或写明不计入",
            "  refuse    负样本（黄金集标 negative: true）期望拒答。判据：答案整段不超 %d 字"
            "且含「无法确定 / 未找到 / 不能提供」等拒答措辞 → 明确拒答；"
            "长篇里夹带一句拒答（先拒后硬答）与空答案都不算 —— 负样本不参与上面三项的分母"
            % _REFUSAL_MAX_CHARS,
            "",
        ]
        for x in self.items:
            if x.negative:
                lines.append("[%s] Q:%s | 负样本(期望拒答) | %s"
                             % ("REFUSED" if x.refused else "ANSWERED", x.question,
                                "已明确拒答" if x.refused else "未拒答（多为用模型自身知识硬答）"))
                lines.append("    答案前90字: %s" % x.answer[:90].replace(chr(10), " / "))
                continue
            tail = (" | 页码:%s->%s" % (x.expect_page, "✓" if x.page_hit else x.pages)
                    if x.has_page else " | 未声明页码; 来源页:%s" % (x.pages,))
            lines.append("[%s] Q:%s | 期望:%s | 命中:%s|忠实:%s%s"
                         % ("PASS" if x.fact_hit else "FAIL",
                            x.question, x.expect or "(未写期望事实)",
                            x.fact_hit, x.grounded, tail))
            lines.append("    答案前90字: %s" % x.answer[:90].replace(chr(10), " / "))
        lines.append("")
        miss = ("；其中 %d 条黄金集条目未写期望事实，按未命中计" % self.missing_expect_count
                if self.missing_expect_count else "")
        neg_note = ("，另有 %d 条负样本另计拒答率" % len(self.negatives)) if self.negatives else ""
        lines.append("结果: 答案含期望事实 %d%%  (%d/%d)%s%s"
                     % (round(self.fact_rate * 100),
                        sum(1 for x in self.positives if x.fact_hit), len(self.positives),
                        miss, neg_note))
        # 拒答率紧挨事实命中率并列 —— 免得「拒答率高是因为什么都不答」被误读
        if self.negatives:
            n_ref = self.refused_count
            lines.append("拒答率(负样本) %d%%  (%d/%d；明确拒答 %d，未拒答 %d)"
                         % (round(self.refuse_rate * 100), n_ref, len(self.negatives),
                            n_ref, len(self.negatives) - n_ref))
        else:
            lines.append("拒答率 不适用  (本次黄金集没有负样本条目)")
        lines.append("引用忠实度(期望事实在随答案返回的来源里) %d%%  (%d/%d)"
                     % (round(self.grounded_rate * 100),
                        sum(1 for x in self.positives if x.grounded), len(self.positives)))
        # 页码这一项无论有没有分母都要出一行 —— 三个数字不能有一个凭空消失
        scored = [x for x in self.positives if x.has_page]
        if scored:
            extra = ("；另有 %d 条黄金集条目未声明页码，不计入" % self.undeclared_page_count
                     if self.undeclared_page_count else "")
            lines.append("引用页码正确 %d%%  (%d/%d)%s"
                         % (round(self.page_rate * 100),
                            sum(1 for x in scored if x.page_hit), len(scored), extra))
        else:
            lines.append("引用页码正确 不适用  (%d 条黄金集条目，无一声明页码，分母为 0)" % self.total)
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
            negative=bool(g.get("negative")), refused=is_refusal(answer),
        ))
        if judge_fn is not None:
            report.items[-1].judged = judge_fn(question, answer, sources)
    return report
