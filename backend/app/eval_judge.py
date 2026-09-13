"""RAGAS 四项的裁判（票 05）—— 按 RAGAS 官方定义算，不是「让模型一次打四个分」。

评测核心不调模型；真正调模型的都在这里，且 LLM 与嵌入从外部注入，
所以测试里换成 stub 就能确定性、无网络地跑。裁判口径固定为 judge 模型 + 温度 0
（见 config.ragas_judge_model / ragas_judge_temperature），报告里会写明，数字才可跨时间比较。

**布尔判定是「一条一问」**（#53）：早先是「一次问 N 条、要求回长度恰好 N 的数组」，
而**让模型自己数数不可靠** —— 真机上 10 条里 9 条因为「回了 5 个、要 6 个」被整条丢掉，
RAGAS 数字只剩 1 个样本。改成一条一问后，长度对不上**结构上不可能发生**；
代价是裁判调用次数 ×N（离线评测可接受）。

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
                 n_questions: int = DEFAULT_N_QUESTIONS):
        if getattr(llm, "api_key", None) == "":
            raise JudgeUnavailable("未配置 LLM API Key，RAGAS 裁判不可用")
        s = get_settings()
        self._llm = llm
        self._embedding = embedding
        self._n = n_questions
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

    def _faithfulness(self, answer: str, contexts: list) -> float:
        return self._supported_ratio(lines_of(self._ask(_STATEMENTS_PROMPT % answer)), contexts)

    def _answer_relevancy(self, question: str, answer: str) -> float:
        generated = lines_of(self._ask(_GENQ_PROMPT % (self._n, answer)))
        if not generated:
            return 0.0
        qv = self._embedding.encode([question])[0]
        return sum(cosine(qv, v) for v in self._embedding.encode(generated)) / len(generated)

    def _context_precision(self, question: str, contexts: list) -> float:
        if not contexts:
            return 0.0
        useful = [self._one_bool(_CTX_REL_ONE_PROMPT % (question, c), "这段上下文有没有用")
                  for c in contexts]
        hits, acc = 0, 0.0
        for k, ok in enumerate(useful, start=1):
            if ok:
                hits += 1
                acc += hits / k      # precision@k，再对所有「有用」的块取平均（RAGAS 口径）
        return acc / hits if hits else 0.0

    def _context_recall(self, reference: str, contexts: list) -> float:
        if not str(reference or "").strip():
            return 0.0               # 没给参考答案就没法算 context_recall，如实报 0
        return self._supported_ratio(lines_of(self._ask(_STATEMENTS_PROMPT % reference)), contexts)

    def _supported_ratio(self, claims: list, contexts: list) -> float:
        if not claims:
            return 0.0
        flags = [self._one_bool(_SUPPORT_ONE_PROMPT % (self._join(contexts), c), "这条论断能否推出")
                 for c in claims]
        return sum(1 for f in flags if f) / len(claims)

    # ---- 与模型打交道 ----

    def _ask(self, prompt: str) -> str:
        try:
            return "".join(self._llm.stream([{"role": "user", "content": prompt}]))
        except Exception as e:   # noqa: BLE001 —— 裁判的任何失败都转成「不可用」，绝不静默
            raise JudgeUnavailable("裁判调用失败：%s: %s" % (type(e).__name__, e))

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
