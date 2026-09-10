"""上下文预算装配：给定历史与预算，**未超预算原样透传**，超了才压缩。

token 计数与摘要都从外部注入 —— 测试因而无网络、无真实 LLM。

本模块只负责「装配」（票 17）。以下**不在**本模块范围：
滚动摘要的持久化（票 18）、真实分词器（票 19）、
枚举意图 / 已引用来源的豁免（票 20-21）。
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
    """token 计数能力（从外部注入；真实分词器见票据 19）。"""

    @abstractmethod
    def count(self, text: str) -> int: ...


class ApproxTokenCounter(TokenCounter):
    """确定性近似：CJK 按字、其余按空白切词。

    **不是**真实分词器，只用于预算装配的取舍判断；对外报数一律用真实分词器。
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
    """用给定 LLM 做摘要。构造参数即工厂：`LlmSummarizer(llm)`。"""

    def __init__(self, llm: LLM):
        self._llm = llm

    def summarize(self, messages: Sequence[dict]) -> str:
        text = "\n".join(format_turn(m) for m in messages)
        return "".join(self._llm.stream([{"role": "user", "content": _SUMMARY_PROMPT + text}])).strip()


@dataclass
class ContextPlan:
    """装配结果：可选摘要 + 保留的近期原文 + 被压掉的轮数。"""

    summary: str | None
    kept: list[dict]
    dropped: int = 0


def _unstripped(history: Sequence[dict] | None) -> list[dict]:
    """原样透传的装配结果（未超预算 / 摘要失败 / 摘要为空时都用它）。"""
    return list(history or [])


def assemble_context(
    history: Sequence[dict] | None,
    *,
    budget: int,
    keep_recent: int,
    count_tokens: Callable[[str], int],
    summarize: Callable[[Sequence[dict]], str],
) -> ContextPlan:
    """把历史装进预算。

    - 轮数不超过 keep_recent：没有"更早的"可压 → 原样透传，**不调 LLM**。
    - 未超预算：原样透传，**不调 LLM**。
    - 超预算：摘要更早的对话，保留最近 keep_recent 轮原文。

    摘要失败或摘要为空时一律**退回原样**——宁可多占预算，也不静默丢掉更早的对话。
    """
    hist = list(history or [])
    if len(hist) <= keep_recent:
        return ContextPlan(summary=None, kept=hist)
    if sum(count_tokens(format_turn(t)) for t in hist) <= budget:
        return ContextPlan(summary=None, kept=hist)

    older, recent = hist[:-keep_recent], hist[-keep_recent:]
    try:
        summary = (summarize(older) or "").strip()
    except Exception as e:  # 摘要器可能因网络/配额失败 —— 降级而非让整个回答崩掉
        logger.warning("对话摘要失败，退回不压缩：%s", e)
        return ContextPlan(summary=None, kept=_unstripped(history))
    if not summary:
        return ContextPlan(summary=None, kept=_unstripped(history))
    return ContextPlan(summary=summary, kept=recent, dropped=len(older))
