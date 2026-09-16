"""RAGAS 四项的裁判（票 05）—— 按 RAGAS 官方定义算，不是「让模型一次打四个分」。

评测核心不调模型；真正调模型的都在这里，且 LLM 与嵌入从外部注入，
所以测试里换成 stub 就能确定性、无网络地跑。裁判口径固定为 judge 模型 + 温度 0
（见 config.ragas_judge_model / ragas_judge_temperature），报告里会写明，数字才可跨时间比较。

**布尔判定是「一条一问」**（#53）：早先是「一次问 N 条、要求回长度恰好 N 的数组」，
而**让模型自己数数不可靠** —— 真机上 10 条里 9 条因为「回了 5 个、要 6 个」被整条丢掉，
RAGAS 数字只剩 1 个样本。改成一条一问后，长度对不上**结构上不可能发生**；
代价是裁判调用次数 ×N —— 所以那些彼此独立的判定**并行跑**（`_bool_each`，并发度可配，设 1 退回串行）。

四项：
  faithfulness      答案的论断能否被检索上下文推出：先拆论断，再逐条核验
  answer_relevancy  从答案反生成问题，与原问题的嵌入相似度取均值
  context_precision 逐块判「对回答这题有没有用」，再按「有用的块是否排在前面」算平均精度
  context_recall    参考答案的论断有多少能从检索上下文里找到

参考答案取自黄金集条目的 reference，没写就退回 expect（一个短事实）；
expect 若只是几个关键词，这一项会退化成 0/1 —— 报告口径里写明了。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings
from app.core.similarity import cosine
from app.utils.text import lines_of

DEFAULT_N_QUESTIONS = 3


class JudgeUnavailable(RuntimeError):
    """裁判不可用（没配 API Key、回复解析不了、条数对不上）。

    必须**显式报错**：绝不能让它悄悄退化成"没算"或给出看着正常的假数字。
    """


_STATEMENTS_PROMPT = (
    "把下面这段文字拆成若干条**互相独立**的论断，一行一条：只输出论断本身，"
    "不要编号、不要解释、不要把两条并成一条。只有一条就只输出一条。\n\n文字：\n%s"
)

# **一条一问**：以前是「一次问 N 条、要求回长度恰好为 N 的数组」，而**让模型自己数数不可靠** ——
# 真机上 10 条里 9 条因为「回了 5 个、要 6 个」被整条丢掉，RAGAS 数字只剩 1 个样本（#53）。
# 一次问一条，n 恒为 1，长度对不上这一类失败**结构上不可能发生**。代价是调用次数 ×N（离线评测可接受）。
_SUPPORT_ONE_PROMPT = (
    "下面这**一条**论断，能否**仅凭**【上下文】推出？能则回 true，不能则回 false。"
    "只回一个词（true 或 false），不要解释、不要输出 JSON。\n"
    "\n【上下文】\n%s\n\n【论断】\n%s"
)

_GENQ_PROMPT = (
    "根据下面这个**答案**，反推出 %d 个它会回答的问题。一行一个：只输出问题本身。\n"
    "\n答案：\n%s"
)

_CTX_REL_ONE_PROMPT = (
    "下面这**一段**【上下文】，对回答【问题】有没有用？有用回 true，没用回 false。"
    "只回一个词（true 或 false），不要解释、不要输出 JSON。\n"
    "\n【问题】\n%s\n\n【上下文】\n%s"
)


# 认布尔：**只认「本身就是判断」的短回复**，认不出就返回 None（调用方按不可用处理）。
#
# 为什么不「在整句里找关键词」：那样会**猜出结论** ——
#   「不确定是否相关」含「否」→ 被判成 false；「not true」含 true → 被判成 true。
# 而这一票的规矩是「解析不出来就报不可用，绝不猜」。提示词已经明确要求「只回一个词」，
# 所以这里严格认，多话的回复就当作没答。
_TRUE_WORDS = frozenset(("true", "yes", "是", "是的", "对", "对的", "可以", "能", "支持", "相关", "有用", "成立"))
_FALSE_WORDS = frozenset(("false", "no", "不是", "否", "不能", "不可以", "无法", "不支持", "不相关",
                          "没用", "无关", "不成立"))
_PUNCT_RE = re.compile(r'''[\s。，,.!！?？:：;；'"`*\\[\\]（）()]+''')


def parse_bool(raw) -> bool | None:
    """认出一个布尔；**认不出返回 None**（调用方按「不可用」处理，绝不猜）。"""
    core = _PUNCT_RE.sub("", str(raw or "").lower())
    if not core or len(core) > 12:      # 这么长不是在回答「能/不能」，不猜
        return None
    if core in _FALSE_WORDS:
        return False
    if core in _TRUE_WORDS:
        return True
    return None


class RagasJudge:
    """RAGAS 四项裁判。构造时注入 LLM 与嵌入；调用一次算一条问答。"""

    def __init__(self, llm, embedding, label: str | None = None,
                 n_questions: int = DEFAULT_N_QUESTIONS,
                 concurrency: int | None = None):
        if getattr(llm, "api_key", None) == "":
            raise JudgeUnavailable("未配置 LLM API Key，RAGAS 裁判不可用")
        s = get_settings()
        self._llm = llm
        self._embedding = embedding
        self._n = n_questions
        # 逐条判定彼此独立 → 并行跑（#55）。设 1 退回串行。
        self._workers = concurrency if concurrency is not None else s.ragas_judge_concurrency
        # 口径字符串要与真正在用的裁判对齐 —— 模型与温度都优先读对象自身的，读不到才回落配置
        model = getattr(llm, "model", None) or s.ragas_judge_model
        temp = getattr(llm, "temperature", None)
        if temp is None:
            temp = s.ragas_judge_temperature
        self.label = label or "RAGAS(judge=%s, temp=%g)" % (model, temp)

    def __call__(self, question: str, answer: str, sources: list, reference: str = "") -> dict:
        contexts = [str(s.get("text", "")) for s in sources if isinstance(s, dict)]
        contexts = [c for c in contexts if c.strip()]
        return {
            "faithfulness": self._faithfulness(answer, contexts),
            "answer_relevancy": self._answer_relevancy(question, answer),
            "context_precision": self._context_precision(question, contexts),
            "context_recall": self._context_recall(reference, contexts),
        }

    # ---- 四项 ----

    def _statements_of(self, text: str) -> list:
        """让裁判把一段文字拆成论断。

        **拆不出来不算 0 分**：0 分是「一条都推不出来」这个**结论**，而空列表是
        「压根没拆出来」（裁判拒答 / 被截断 / 回复为空）。两者混起来，就会把
        「裁判挂了」印成 [未达标] —— 而 RAGAS 忠实度是报告里的一条目标线（#64 批 4）。
        """
        got = lines_of(self._ask(_STATEMENTS_PROMPT % text))
        if not got and str(text or "").strip():
            raise JudgeUnavailable("裁判没有把这段文字拆成任何论断（回复为空或被截断）")
        return got

    def _faithfulness(self, answer: str, contexts: list) -> float:
        return self._supported_ratio(self._statements_of(answer), contexts)

    def _answer_relevancy(self, question: str, answer: str) -> float:
        generated = lines_of(self._ask(_GENQ_PROMPT % (self._n, answer)))
        if not generated:
            if str(answer or "").strip():
                raise JudgeUnavailable("裁判没有从答案里反生成出任何问题（回复为空或被截断）")
            return 0.0
        qv = self._embedding.encode([question])[0]
        return sum(cosine(qv, v) for v in self._embedding.encode(generated)) / len(generated)

    def _context_precision(self, question: str, contexts: list) -> float:
        if not contexts:
            return 0.0
        useful = self._bool_each([_CTX_REL_ONE_PROMPT % (question, c) for c in contexts],
                                 "这段上下文有没有用")
        hits, acc = 0, 0.0
        for k, ok in enumerate(useful, start=1):
            if ok:
                hits += 1
                acc += hits / k      # precision@k，再对所有「有用」的块取平均（RAGAS 口径）
        return acc / hits if hits else 0.0

    def _context_recall(self, reference: str, contexts: list) -> float:
        if not str(reference or "").strip():
            return 0.0               # 没给参考答案就没法算 context_recall，如实报 0
        return self._supported_ratio(self._statements_of(reference), contexts)

    def _supported_ratio(self, claims: list, contexts: list) -> float:
        if not claims:
            return 0.0
        flags = self._bool_each([_SUPPORT_ONE_PROMPT % (self._join(contexts), c) for c in claims],
                                "这条论断能否推出")
        return sum(1 for f in flags if f) / len(claims)

    # ---- 与模型打交道 ----

    def _ask(self, prompt: str) -> str:
        try:
            return "".join(self._llm.stream([{"role": "user", "content": prompt}]))
        except Exception as e:   # noqa: BLE001 —— 裁判的任何失败都转成「不可用」，绝不静默
            raise JudgeUnavailable("裁判调用失败：%s: %s" % (type(e).__name__, e))

    def _bool_each(self, prompts: list, what: str) -> list:
        """逐条判定 —— **并行**跑（#55）。

        一条一问（#53）把「模型数不准就整条打回」换成了「一条一次」，代价是调用次数 ×N。
        但这些判定**彼此完全独立**，串行等网络就是白等；`Executor.map` 保序，
        结果与入参一一对应。并发度来自配置 —— 设 1 就退回串行。
        """
        if not prompts:
            return []
        workers = min(self._workers, len(prompts))
        if workers <= 1:
            return [self._one_bool(p, what) for p in prompts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda p: self._one_bool(p, what), prompts))

    def _one_bool(self, prompt: str, what: str) -> bool:
        """问**一条**、回**一个**布尔。认不出来就报不可用 —— 绝不猜。"""
        raw = self._ask(prompt)
        got = parse_bool(raw)
        if got is None:
            raise JudgeUnavailable("裁判没给出可判定的答复（%s），实回：%r" % (what, raw[:120]))
        return got

    @staticmethod
    def _join(contexts: list) -> str:
        return "\n\n".join(contexts) if contexts else "(无上下文)"
