"""RAGAS 四项的裁判（票 05）—— 按 RAGAS 官方定义算，不是「让模型一次打四个分」。

评测核心不调模型；真正调模型的都在这里，且 LLM 与嵌入从外部注入，
所以测试里换成 stub 就能确定性、无网络地跑。裁判口径固定为 judge 模型 + 温度 0
（见 config.ragas_judge_model / ragas_judge_temperature），报告里会写明，数字才可跨时间比较。

四项：
  faithfulness      答案的论断能否被检索上下文推出：先拆论断，再逐条核验
  answer_relevancy  从答案反生成问题，与原问题的嵌入相似度取均值
  context_precision 逐块判「对回答这题有没有用」，再按「有用的块是否排在前面」算平均精度
  context_recall    参考答案的论断有多少能从检索上下文里找到

参考答案取自黄金集条目的 reference，没写就退回 expect（一个短事实）；
expect 若只是几个关键词，这一项会退化成 0/1 —— 报告口径里写明了。
"""
from __future__ import annotations

from app.config import get_settings
from app.core.similarity import cosine
from app.utils.text import extract_json, lines_of

DEFAULT_N_QUESTIONS = 3


class JudgeUnavailable(RuntimeError):
    """裁判不可用（没配 API Key、回复解析不了、条数对不上）。

    必须**显式报错**：绝不能让它悄悄退化成"没算"或给出看着正常的假数字。
    """


_STATEMENTS_PROMPT = (
    "把下面这段文字拆成若干条**互相独立**的论断，一行一条：只输出论断本身，"
    "不要编号、不要解释、不要把两条并成一条。只有一条就只输出一条。\n\n文字：\n%s"
)

_SUPPORT_PROMPT = (
    "下面每条论断，能否**仅凭**【上下文】推出？逐条判断：是则 true，否则 false。"
    "只输出一个 JSON 数组（长度与论断条数一致，元素是 true / false），不要任何解释。\n"
    "\n【上下文】\n%s\n\n【论断】\n%s\n\nJSON 数组："
)

_GENQ_PROMPT = (
    "根据下面这个**答案**，反推出 %d 个它会回答的问题。一行一个：只输出问题本身。\n"
    "\n答案：\n%s"
)

_CTX_REL_PROMPT = (
    "下面每一段【上下文】，对回答【问题】有没有用？逐段判断：有用则 true，否则 false。"
    "只输出一个 JSON 数组（长度与段数一致，元素是 true / false），不要任何解释。\n"
    "\n【问题】\n%s\n\n【上下文】\n%s\n\nJSON 数组："
)


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
        useful = self._bools(_CTX_REL_PROMPT % (question, self._join(contexts)), len(contexts))
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
        flags = self._bools(_SUPPORT_PROMPT % (self._join(contexts), "\n".join(claims)),
                            len(claims))
        return sum(1 for f in flags if f) / len(claims)

    # ---- 与模型打交道 ----

    def _ask(self, prompt: str) -> str:
        try:
            return "".join(self._llm.stream([{"role": "user", "content": prompt}]))
        except Exception as e:   # noqa: BLE001 —— 裁判的任何失败都转成「不可用」，绝不静默
            raise JudgeUnavailable("裁判调用失败：%s: %s" % (type(e).__name__, e))

    def _bools(self, prompt: str, n: int) -> list:
        raw = self._ask(prompt)
        data = extract_json(raw, "[")
        data = data if isinstance(data, list) else None
        if data is None or len(data) != n:
            raise JudgeUnavailable("裁判没有按要求回 %d 个 true/false，实回：%r" % (n, raw[:120]))
        return [bool(x) for x in data]

    @staticmethod
    def _join(contexts: list) -> str:
        return "\n\n".join(contexts) if contexts else "(无上下文)"
