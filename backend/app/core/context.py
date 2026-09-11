"""上下文预算装配：给定历史与预算，**未超预算原样透传**，超了才压缩。

token 计数与摘要都从外部注入 —— 测试因而无网络、无真实 LLM。

票 18 起支持**滚动摘要**：给一份既有摘要与它的游标，只滚动**未覆盖的尾部** ——
尾部装得下就复用旧摘要（不调 LLM），尾部又超预算才并入摘要、游标前进。
持久化（写回会话）在调用方（chat_service）；本模块只管装配与游标推进。

以下**不在**本模块范围：真实分词器（票 19）、枚举意图 / 已引用来源的豁免（票 20-21）。
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from app.core.llm import LLM
from app.core.prompt import format_turn

logger = logging.getLogger(__name__)

_CJK = re.compile(r"[㐀-䶿一-鿿]")


class TokenCounter(ABC):
    """token 计数能力（从外部注入；真实分词器见票据 19）。

    `label` 是**对外报数的口径名**：空串表示「这不是真实分词器」—— 此时任何 token
    指标都必须报「不可用」，绝不许拿估算值顶替（spec 0003 的降级诚实性）。
    `note` 写清为什么没有真实分词器。
    """

    label: str = ""
    note: str = ""

    @abstractmethod
    def count(self, text: str) -> int: ...


class ApproxTokenCounter(TokenCounter):
    """确定性近似：CJK 按字、其余按空白切词。

    **不是**真实分词器（`label` 为空），只用于预算装配的取舍判断；对外报数一律用真实分词器。
    """

    def count(self, text: str) -> int:
        s = text or ""
        cjk = len(_CJK.findall(s))
        words = len([w for w in re.split(r"\s+", _CJK.sub(" ", s)) if w])
        return cjk + words


class Summarizer(ABC):
    """把一段更早的对话压成摘要（从外部注入）。"""

    @abstractmethod
    def summarize(self, messages: Sequence[dict]) -> str: ...


_SUMMARY_PROMPT = (
    "把下面这段更早的对话压成一段简短摘要，保留事实、结论与用户偏好。"
    "只输出摘要正文，不要客套话。\n\n"
)


class LlmSummarizer(Summarizer):
    """用给定 LLM 做摘要。构造参数即工厂：`LlmSummarizer(llm)`。

    演示/测试的确定性由注入的 LLM 保证（FakeLLM 返回固定文本），不必另造一个 FakeSummarizer。
    """

    def __init__(self, llm: LLM):
        self._llm = llm

    def summarize(self, messages: Sequence[dict]) -> str:
        text = "\n".join(format_turn(m) for m in messages)
        return "".join(self._llm.stream([{"role": "user", "content": _SUMMARY_PROMPT + text}])).strip()


@dataclass
class ContextPlan:
    """装配结果：可选摘要 + 保留的近期原文 + 被压掉的轮数 + 摘要覆盖到哪（游标）。"""

    summary: str | None
    kept: list[dict]
    dropped: int = 0
    cursor: str | None = None    # 摘要已覆盖到的最后一条消息 id（票 18 的持久化游标）


_PREVIOUS_LABEL = "以下是此前对话的摘要"


def _last_id(turn: dict) -> str | None:
    """这一轮的最后一条消息 id（游标单位）—— 没有 id 的历史（旧调用方）返回 None。"""
    ids = turn.get("ids") or []
    return ids[-1] if ids else None


def _covered_turns(history: Sequence[dict], upto: str | None) -> int:
    """前多少轮已被既有摘要覆盖。

    游标不在窗口里（早就掉出「最近 N 条」）-> 窗口内一轮都没覆盖：这时的尾部就是
    整段窗口，与「从没摘要过」等价 —— 宁可多算一次，也不丢对话。
    """
    if not upto:
        return 0
    for i, turn in enumerate(history):
        if _last_id(turn) == upto:
            return i + 1
    return 0


def _with_previous(turns: Sequence[dict], previous: str) -> list[dict]:
    """把旧摘要摆在最前面：新一轮摘要是**累计**的，不会把更早的对话丢掉一遍。"""
    if not previous:
        return list(turns)
    return [{"user": _PREVIOUS_LABEL, "assistant": previous}] + list(turns)


def assemble_context(
    history: Sequence[dict] | None,
    *,
    budget: int,
    keep_recent: int,
    count_tokens: Callable[[str], int],
    summarize: Callable[[Sequence[dict]], str],
    previous: str | None = None,
    previous_upto: str | None = None,
) -> ContextPlan:
    """把历史装进预算；有既有摘要（票 18）时只滚动**未覆盖的尾部**。

    - 未摘要的尾部装得下（或本来就没几轮）：复用旧摘要，**不调 LLM**。
    - 尾部也超预算：把尾部更早的部分并入摘要（摘要器能看到旧摘要），游标前进。
    - 没有既有摘要时与票 17 完全一致：未超预算原样透传，超了才压一次。

    摘要失败或摘要为空时一律**退回不压缩**——宁可多占预算，也不静默丢掉更早的对话。
    预算只衡量**未摘要的尾部**；摘要本身占的额度不计（票 19 报降幅时会一并算）。
    """
    hist = list(history or [])
    kept_summary = previous or None
    # 游标只在**确实有摘要**时才算数：摘要空而游标在，会让被覆盖的轮既没原文也没摘要
    cursor = previous_upto if kept_summary else None
    tail = hist[_covered_turns(hist, cursor):]
    # 复用：尾部没超预算就没必要再压 —— 游标原地不动，也不用调摘要器
    if len(tail) <= keep_recent:
        return ContextPlan(summary=kept_summary, kept=tail, cursor=cursor)
    if sum(count_tokens(format_turn(t)) for t in tail) <= budget:
        return ContextPlan(summary=kept_summary, kept=tail, cursor=cursor)

    older, recent = tail[:-keep_recent], tail[-keep_recent:]
    try:
        summary = (summarize(_with_previous(older, previous)) or "").strip()
    except Exception as e:  # 摘要器可能因网络/配额失败 —— 降级而非让整个回答崩掉
        logger.warning("对话摘要失败，退回不压缩：%s", e)
        return ContextPlan(summary=kept_summary, kept=tail, cursor=cursor)
    if not summary:
        return ContextPlan(summary=kept_summary, kept=tail, cursor=cursor)
    return ContextPlan(summary=summary, kept=recent, dropped=len(older),
                       cursor=_last_id(older[-1]) or cursor)
