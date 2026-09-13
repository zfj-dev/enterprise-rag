"""评测核心（票 01）：注入式纯函数 —— 给任意「问答实现」算出一份确定的评测数字。

`run_eval(黄金集, answer_fn, judge_fn)` 的**被评对象与裁判都从外部注入**，所以核心
**不连服务、不调 LLM**：离线、CI、测试里都能跑出确定结果，换个后端也不用改评测代码。

判据口径（免 LLM）：把期望事实与待查文本都「去空白 + 转小写」后做子串匹配 ——
解析器会在数字/标点之间插空格（'表 3 . 1'、'Windows 11'），不这样归一化会整片漏判。
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Callable, Sequence

# 问题 -> {"answer": str, "sources": [{"text": str, "page": int, ...}]}
AnswerFn = Callable[[str], dict]
# (问题, 答案, 来源, 参考答案) -> 裁判结论（RAGAS 四项见票 05 与 app/eval_judge.py）
JudgeFn = Callable[[str, str, list, str], dict]

# RAGAS 四项的固定指标名 —— 裁判返回的 dict 就按这几个键取值汇总
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def as_int(value) -> int | None:
    """把外来的 token 数收成 int —— bool 也是 int，别把 True 当 1 个 token。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def embedding_label() -> str:
    """报告里写「用的是哪个嵌入」—— **读配置，不读环境变量**，而且**要带上模型名**。

    坑一：`.env` 里的值**不会**进 `os.environ`（pydantic 只把它灌进 Settings），
    照环境变量渲染会把自己写成「fake」—— 真机上跑的是 bge-m3，报告却写「嵌入: fake」（#52）。
    坑二：只写 provider 不够 —— 换模型就换了口径，报告得让人看出**是哪个模型**。
    """
    from app.config import get_settings

    s = get_settings()
    if s.embedding_provider == "fake":
        return "fake（字符词袋，**不是真模型**）"      # 不写成 bge-xxx，免得看着像真跑了
    return "%s / %s" % (s.embedding_provider, s.embedding_model)


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
    group: str = ""             # 分组名（如 RGB 的四能力）；空则不参与「分能力」汇总
    negative: bool = False      # 负样本：答案不在文档里，期望拒答（黄金集写 "negative": true）
    citation_coverage: float | None = None  # 引用覆盖率（逐句核验）；问答实现没给就是 None
    ctx_before: int | None = None           # 压缩前 token（**真实分词器**数的；没有就是 None）
    ctx_after: int | None = None            # 压缩后 token
    ctx_tokenizer: str = ""                 # token 口径（哪个分词器）；空 = 没有真实分词器
    ctx_note: str = ""                      # 没有真实分词器时的原因
    ctx_budget: int | None = None            # 当时的上下文预算（口径三件套之一）
    refused: bool = False       # 判据认定「明确拒答」
    judged: dict | None = None  # 注入 judge_fn 时的裁判结论
    judge_error: str | None = None  # 这条裁判挂了的原因（要明说，不能当没算过）

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
    retrieval: RetrievalMetrics | None = None   # 检索层指标（票 04）；有就一并渲染（票 08 接线）
    latency: LatencyMetrics | None = None       # 延迟分桶（票 07）；同上
    judge_label: str | None = None              # 裁判口径，写进报告才可跨时间比较
    judge_error: str | None = None              # 裁判整体不可用的原因（缺 Key 等）
    # 重排降级/未评分的次数（票 39）：降级不打断回答是对的，但报告得说清这批数字
    # 是不是在「没重排」的情况下跑出来的。
    rerank_degraded: int = 0
    rerank_unscored: int = 0

    @property
    def ragas(self) -> dict | None:
        """RAGAS 四项的均值；一条都没打上分则为 None（报告会说清为什么）。"""
        scored = [x.judged for x in self.items if isinstance(x.judged, dict)]
        if not scored:
            return None
        out: dict = {}
        for metric in RAGAS_METRICS:
            vals = [float(d[metric]) for d in scored
                    if isinstance(d.get(metric), (int, float))]
            if vals:
                out[metric] = sum(vals) / len(vals)
        return out or None

    @property
    def ragas_count(self) -> int:
        """打上分的**正样本**条数 —— 与正样本总数一起报，缺几条一眼看得出。

        分母只算正样本：负样本走拒答率，不进 RAGAS。
        """
        return sum(1 for x in self.positives if isinstance(x.judged, dict))

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def group_names(self) -> list[str]:
        """出现过哪些分组，按首次出现排序。"""
        names: list[str] = []
        for x in self.items:
            if x.group and x.group not in names:
                names.append(x.group)
        return names

    def sub(self, group: str) -> "Report":
        """某个分组的子报告 —— 事实命中 / 拒答率等口径完全复用，不另写一套公式。"""
        return Report(items=[x for x in self.items if x.group == group])

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
    def citation_coverage_rate(self) -> float | None:
        """引用覆盖率均值（只算拿到这项数据的条目）；一条都没有则为 None。"""
        vals = [float(x.citation_coverage) for x in self.positives
                if isinstance(x.citation_coverage, (int, float))]
        return sum(vals) / len(vals) if vals else None

    @property
    def citation_coverage_count(self) -> int:
        """有多少条真的拿到了引用覆盖率。"""
        return sum(1 for x in self.positives
                   if isinstance(x.citation_coverage, (int, float)))

    @property
    def refuse_rate(self) -> float | None:
        """负样本里「明确拒答」的占比；没有负样本条目则为 None。"""
        negs = self.negatives
        if not negs:
            return None
        return self.refused_count / len(negs)

    @property
    def token_pairs(self) -> list:
        """可用于算降幅的 (压缩前, 压缩后) —— 压缩前为 0 的条目不算（降幅没有意义）。

        三条口径（降幅 / 条数 / 报告里的求和）**共用这一处**，免得各筛各的、数字对不上。
        """
        return [(x.ctx_before, x.ctx_after) for x in self.positives
                if as_int(x.ctx_before) and as_int(x.ctx_after) is not None and x.ctx_before > 0]

    @property
    def token_reduction_rate(self) -> float | None:
        """上下文压缩降幅 = 1 - Σ压缩后/Σ压缩前；没有可算的条目就是 None。

        **只用真实分词器报出来的数**：没有就返回 None，报告那边写「不可用」——
        绝不拿字符估算顶替（spec 0003 的降级诚实性）。
        """
        pairs = self.token_pairs
        if not pairs:
            return None
        before = sum(b for b, _ in pairs)
        after = sum(a for _, a in pairs)
        return 1 - after / before

    @property
    def token_reduction_count(self) -> int:
        """有多少条真的算进了降幅。"""
        return len(self.token_pairs)

    @property
    def tokenizer_label(self) -> str:
        """token 口径（哪个分词器）—— 报告里必须写出来，数字脱离口径就不可信。"""
        for x in self.items:
            if x.ctx_tokenizer:
                return x.ctx_tokenizer
        return ""

    @property
    def tokenizer_note(self) -> str:
        """没有真实分词器时的原因（取第一条说清楚就够）。"""
        for x in self.items:
            if x.ctx_note:
                return x.ctx_note
        return ""

    @property
    def reduction_missing_reason(self) -> str:
        """没有降幅数字时的**原因**（只此一处）——「有分词器但没可压的历史」与「没分词器」是两回事。

        三处渲染（本模块的 `_token_lines`、`evaluate_all`、`eval_compare`）都从这里取原因：
        各写各的必然会漂移，把「不适用」说成「没有真实分词器」就成了假话。
        """
        if self.tokenizer_label:
            return "不适用（本次没有可压的多轮历史；口径 %s）" % self.tokenizer_label
        return "不可用（%s）" % (self.tokenizer_note or "没有真实分词器（不拿字数估算顶替）")

    def _token_lines(self) -> list:
        """压缩降幅那一段 —— 三种情形分得清清楚楚，绝不把「没数据」说成「没分词器」。

        口径三件套一起写：**分词器 / 预算 / 触发条件**（数字脱离口径就不可信）。
        """
        red = self.token_reduction_rate
        budget = next((x.ctx_budget for x in self.positives if as_int(x.ctx_budget)), None)
        scope = "触发条件: 超预算才压；口径: %s%s%s" % (
            self.tokenizer_label or "（未接真实分词器）",
            "；预算 %d tokens" % budget if budget else "",
            "；只算历史那部分，记忆不计入")
        if red is not None:
            pairs = self.token_pairs
            return ["上下文压缩降幅(token) %d%%  (压缩前 %d -> 压缩后 %d；%d 条计入；%s)"
                    % (round(red * 100), sum(b for b, _ in pairs),
                       sum(a for _, a in pairs), len(pairs), scope)]
        return ["上下文压缩降幅(token) %s  (%s)"
                % (self.reduction_missing_reason, scope)]

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
            "  coverage  引用覆盖率：答案的论断里被来源支撑的占比（逐句核验；"
            "免 LLM 的跑法拿不到就不出这一行）",
            "  黄金集条目缺期望事实 / 页码时：不跳过该条，而是按未命中计入或写明不计入"
            "  token    压缩前 / 压缩后 token 与降幅，用真实分词器数；没有真实分词器就写「不可用」，"
            "有分词器但这轮没有可压的多轮历史就写「不适用」——两种「没数字」不许混成一句",
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
        # 压缩降幅紧挨事实命中率 —— 只报降幅不报质量，等于奖励「把上下文砍掉」
        lines.extend(self._token_lines())
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
        if self.citation_coverage_rate is not None:
            lines.append("引用覆盖率(论断被来源支撑) %d%%  (计入 %d 条)"
                         % (round(self.citation_coverage_rate * 100), self.citation_coverage_count))
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
        if self.group_names:
            lines.append("")
            lines.append("=== 分能力 ===")
            lines.append("按黄金集条目的 group 分组；每组的口径与总表一致 —— "
                         "正样本出事实命中率，负样本出拒答率（没这类条目的格子里是 -）")
            lines.append("%-16s %5s %8s %8s" % ("组", "条数", "事实命中", "拒答率"))
            for name in self.group_names:
                grp = self.sub(name)
                fact = "%.0f%%" % (grp.fact_rate * 100) if grp.positives else "-"
                ref = "%.0f%%" % (grp.refuse_rate * 100) if grp.negatives else "-"
                lines.append("%-16s %5d %8s %8s" % (name, len(grp.items), fact, ref))
        lines.append("")
        lines.append("=== RAGAS 四项 ===")
        if self.ragas is None:
            lines.append("没有可用的 RAGAS 数字：%s"
                         % ("裁判没给出可解析的分" if self.ragas_count else "未接裁判"))
            if self.judge_error:
                lines.append("  裁判不可用：%s" % self.judge_error)
        else:
            lines.append("裁判口径：%s" % (self.judge_label or "(未标注)"))
            lines.append("逐条由裁判按 RAGAS 各指标定义算出 0~1 分，报告取均值；"
                         "计入 %d/%d 条（只算正样本，负样本归拒答率）"
                         % (self.ragas_count, len(self.positives)))
            lines.append("  context_recall 的基准取黄金集条目的 reference，没写就退回 expect；"
                         "expect 若只是几个关键词，这一项会退化成 0/1")
            for metric in RAGAS_METRICS:
                v = self.ragas.get(metric)
                lines.append("  %-18s %s" % (metric, "%.3f" % v if v is not None else "缺"))
            if self.judge_error:
                lines.append("  部分条目裁判失败：%s" % self.judge_error)
        if self.latency is not None:
            lines.append("")
            lines.extend(self.latency.to_lines())
        if self.retrieval is not None or self.latency is not None:
            lines.insert(0, "=== 生成层指标 ===")
        if self.retrieval is not None:
            lines.append("")
            lines.extend(self.retrieval.to_lines())
        return lines


def run_eval(goldenset: Sequence[dict], answer_fn: AnswerFn,
             judge_fn: JudgeFn | None = None,
             judge_label: str | None = None) -> Report:
    """对黄金集逐条跑 answer_fn，按固定口径判据算出报告。

    judge_fn 挂了不会被吞掉：记在条目与报告上，报告里明说 —— 宁可报错也不给假数字。
    """
    report = Report(judge_label=judge_label)
    for g in goldenset:
        question = g.get("question", "")
        expect = g.get("expect", "")
        expect_page = g.get("page")
        out = answer_fn(question) or {}
        answer = out.get("answer", "") or ""
        sources = [s for s in (out.get("sources") or []) if isinstance(s, dict)]
        ctx = out.get("context") if isinstance(out.get("context"), dict) else {}

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
            group=g.get("group", ""),
            negative=bool(g.get("negative")), refused=is_refusal(answer),
            citation_coverage=out.get("citation_coverage"),
            ctx_before=as_int(ctx.get("tokens_before")), ctx_after=as_int(ctx.get("tokens_after")),
            ctx_tokenizer=str(ctx.get("tokenizer") or ""), ctx_note=str(ctx.get("note") or ""),
            ctx_budget=as_int(ctx.get("budget")),
        ))
        # 负样本只判拒答：它本就没有参考答案，送进 RAGAS 只会把四项均值无端拖低
        if judge_fn is not None and not g.get("negative"):
            item = report.items[-1]
            reference = g.get("reference") or expect
            try:
                item.judged = judge_fn(question, answer, sources, reference)
            except Exception as e:   # noqa: BLE001 —— 裁判失败要记下来，不能被当成"没算"
                item.judge_error = "%s: %s" % (type(e).__name__, e)
                if report.judge_error is None:
                    report.judge_error = item.judge_error
    return report


# ---------- 检索层指标（票 04） ----------
# 与生成层同一份报告渲染，但**离线可跑**：检索方式与打分全由 retrieve_fn 注入，
# 核心只做算术 —— 不依赖运行中的服务，也不依赖真实嵌入。

def hit_recall_at_k(ranked_ids: Sequence[str], gold: set, k: int) -> tuple[float, float]:
    """hit@k：任一正确分块落在前 k 条；recall@k：前 k 条里命中的占全部正确分块的比例。"""
    ids = list(ranked_ids)[:k]
    hit = 1.0 if any(g in ids for g in gold) else 0.0
    recall = len(gold & set(ids)) / max(1, len(gold))
    return hit, recall


def reciprocal_rank(ranked_ids: Sequence[str], gold: set) -> float:
    """RR：首个正确分块名次的倒数。**用全排名**，不受 k 影响。"""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


@dataclass
class LabelScore:
    """一种检索方式的成绩单。"""

    hit: dict = field(default_factory=dict)      # k -> 平均 hit@k
    recall: dict = field(default_factory=dict)   # k -> 平均 recall@k
    mrr: float = 0.0


@dataclass
class QuestionRow:
    """逐条明细：这条问题捞到没有 —— 用来定位「检索是否漏了正确答案」。"""

    question: str
    doc: str
    expect: str
    gold_count: int
    rr: dict = field(default_factory=dict)       # 方式名 -> RR


@dataclass
class NegativeCheck:
    """负样本：向量 top-1 相似度越低，越有机会走到「没有可用来源 → 拒答」。"""

    question: str
    top1: float | None
    chunk_id: str | None = None
    below: bool | None = None                    # top1 是否低于参照阈值；没给阈值时为 None


@dataclass
class RetrievalMetrics:
    """多检索方式 × 多 k 的成绩，外加负样本的相似度检查。"""

    ks: tuple
    labels: tuple
    scores: dict = field(default_factory=dict)     # 方式名 -> LabelScore
    scored_count: int = 0                          # 计入指标的正样本条数
    skipped_count: int = 0                         # 判不出正确分块、未计入的条数
    skipped_questions: list = field(default_factory=list)   # 上面这些是哪些题（不静默）
    negatives: list = field(default_factory=list)  # [NegativeCheck]
    rows: list = field(default_factory=list)       # [QuestionRow]
    threshold: float | None = None                 # 参照阈值（见 to_lines 的口径说明）

    def to_lines(self) -> list[str]:
        lines = [
            "=== 检索层指标 ===",
            "判据口径：正确分块 = 内容含期望事实（声明了页码时还要求页码相符）的 child 块",
            "  hit@k    任一正确分块落在前 k 条",
            "  recall@k 前 k 条里命中的正确分块占全部正确分块的比例",
            "  MRR      首个正确分块名次的倒数（用全排名，不看 k）",
            "",
        ]
        lines.append("计入 %d 条正样本" % self.scored_count
                     + ("；%d 条因判不出正确分块未计入：%s"
                        % (self.skipped_count, " / ".join(self.skipped_questions))
                        if self.skipped_count else ""))
        lines.append("")

        head = "%-14s" % "方式"
        for k in self.ks:
            head += " hit@%-2d rec@%-2d " % (k, k)
        lines.append(head + " MRR")
        for label in self.labels:
            sc = self.scores[label]
            row = "%-14s" % label
            for k in self.ks:
                row += " %5.2f  %5.2f " % (sc.hit[k], sc.recall[k])
            lines.append(row + " %.3f" % sc.mrr)

        lines.append("")
        lines.append(self._negative_header())
        if not self.negatives:
            lines.append("  本次没有负样本条目")
        for n in self.negatives:
            if n.below is None:
                tag = "无相似度"
            else:
                tag = "低(该拒)" if n.below else "高(会被当成相关内容)"
            lines.append("  [%s] Q:%s | 向量top-1=%s%s"
                         % (tag, n.question, n.top1,
                            " chunk=%s" % n.chunk_id if n.chunk_id else ""))

        if self.rows:
            lines.append("")
            lines.append("--- 逐条明细（RR = 首个正确分块名次的倒数，全排名；0 = 一个都没捞到）---")
            for r in self.rows:
                marks = "  ".join("%s=%.2f" % (l, r.rr.get(l, 0.0)) for l in self.labels)
                best = max(r.rr.values()) if r.rr else 0.0
                where = "[%s] " % r.doc if r.doc else ""
                lines.append("[%s] %sQ:%s | 期望:%s | 正确分块=%d 条 | %s"
                             % ("OK" if best > 0 else "MISS", where, r.question,
                                r.expect or "(未写)", r.gold_count, marks))
        return lines

    def _negative_header(self) -> str:
        if self.threshold is None:
            return "负样本（期望拒答）：未给参照阈值，只记录向量 top-1 相似度"
        return ("负样本（期望拒答）：向量 top-1 相似度参照阈值 %.2f —— "
                "**这不是系统真正的拒答条件**（系统是「没有可用来源就不作断言」，"
                "并不按相似度卡），这里只看负样本的相似度是否明显偏低。" % self.threshold)


def run_retrieval_eval(questions: Sequence[dict], retrieve_fn: Callable[[str], dict],
                       ks: Sequence[int] = (3, 5, 10),
                       threshold: float | None = None) -> RetrievalMetrics:
    """算检索层指标。**不碰服务、不碰真实嵌入** —— 检索方式与打分全由 retrieve_fn 注入。

    questions 每条：{"question", "doc"?, "expect"?, "gold_ids": 该问题的正确分块 id 集合}；
                    负样本写 {"question", "negative": True}（只查 top-1 相似度）。
    retrieve_fn(question) -> {"ranks": {方式名: [分块 id 按名次]},
                              "top1": 向量 top-1 相似度, "top1_id": 那个分块 id}
    """
    ks = tuple(ks)
    acc: dict[str, dict] = {}          # 方式名 -> {"hit": {k: []}, "recall": {k: []}, "rr": []}

    def bucket(label: str) -> dict:
        return acc.setdefault(label, {"hit": {k: [] for k in ks},
                                      "recall": {k: [] for k in ks}, "rr": []})

    scored_count = 0
    skipped_questions: list[str] = []
    negatives: list[NegativeCheck] = []
    rows: list[QuestionRow] = []

    for q in questions:
        out = retrieve_fn(q["question"]) or {}
        ranks = out.get("ranks") or {}
        for label in ranks:
            bucket(label)

        if q.get("negative"):
            top1 = out.get("top1")
            below = (top1 < threshold) if (top1 is not None and threshold is not None) else None
            negatives.append(NegativeCheck(question=q["question"], top1=top1,
                                           chunk_id=out.get("top1_id"), below=below))
            continue

        gold = set(q.get("gold_ids") or ())
        if not gold:            # 判不出正确分块 —— 记下是哪一题，别让它悄悄把分母改小
            skipped_questions.append(q["question"])
            continue

        scored_count += 1
        row_rr: dict[str, float] = {}
        for label, ranked in ranks.items():
            b = bucket(label)
            for k in ks:
                h, r = hit_recall_at_k(ranked, gold, k)
                b["hit"][k].append(h)
                b["recall"][k].append(r)
            rr = reciprocal_rank(ranked, gold)
            b["rr"].append(rr)
            row_rr[label] = rr
        rows.append(QuestionRow(question=q["question"], doc=q.get("doc", ""),
                                expect=q.get("expect", ""), gold_count=len(gold), rr=row_rr))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return RetrievalMetrics(
        ks=ks, labels=tuple(acc.keys()),
        scores={l: LabelScore(hit={k: avg(v["hit"][k]) for k in ks},
                              recall={k: avg(v["recall"][k]) for k in ks},
                              mrr=avg(v["rr"]))
                for l, v in acc.items()},
        scored_count=scored_count,
        skipped_count=len(skipped_questions),
        skipped_questions=skipped_questions,
        negatives=negatives, rows=rows, threshold=threshold,
    )


# ---------- 延迟（票 07） ----------
# 与其它指标同一份报告渲染；分段与并发都由调用方喂样本，核心只做统计。

LATENCY_STAGES = ("retrieval", "rerank", "ttft", "generate")

_STAGE_LABELS = {"retrieval": "检索", "rerank": "重排", "ttft": "首字(TTFT)", "generate": "生成"}


def percentile(values: Sequence, p: float) -> float | None:
    """最近秩法（nearest-rank）：排序后取第 ceil(p/100 × n) 个。

    样本少时比插值法稳，也不会出现 P95 小于 P50 这种怪事。空样本返回 None。
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = max(1, math.ceil(p / 100 * len(xs)))
    return xs[min(k, len(xs)) - 1]


@dataclass
class LatencyMetrics:
    """并发下的分段耗时：检索 / 重排 / 生成 / 首字，各给 P50 与 P95。"""

    concurrent: int = 1
    rounds: int = 1
    samples: list = field(default_factory=list)   # 每次一条：{阶段名: 毫秒}
    note: str = ""                                # 测量方式说明（由脚本填，写进报告）

    @property
    def count(self) -> int:
        return len(self.samples)

    def p(self, stage: str, pct: float):
        """某一段的第 pct 百分位（毫秒）；没采到就是 None。"""
        return percentile([s.get(stage) for s in self.samples], pct)

    def to_lines(self) -> list[str]:
        lines = [
            "=== 延迟（并发 %d × %d 轮）===" % (self.concurrent, self.rounds),
            "测量方式：%s" % (self.note or "(未标注)"),
            "本段只测延迟，**不参与正确性判定** —— 正确性按单次顺序跑的那一遍判",
            "",
        ]
        if not self.count:
            lines.append("没有采到样本")
            return lines
        lines.append("%-12s %10s %10s" % ("分段", "P50", "P95"))
        for stage in LATENCY_STAGES:
            p50, p95 = self.p(stage, 50), self.p(stage, 95)
            lines.append("%-12s %10s %10s"
                         % (_STAGE_LABELS[stage],
                            "-" if p50 is None else "%.0f ms" % p50,
                            "-" if p95 is None else "%.0f ms" % p95))
        lines.append("样本 n=%d" % self.count)
        return lines
