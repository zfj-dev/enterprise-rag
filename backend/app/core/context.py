"""上下文压缩：会话摘要装配（票 17-20）+ 工具结果清理的纯函数（票 21）。

token 计数与摘要都从外部注入 —— 测试因而无网络、无真实 LLM。

票 18 起支持**滚动摘要**：给一份既有摘要与它的游标，只滚动**未覆盖的尾部** ——
尾部装得下就复用旧摘要（不调 LLM），尾部又超预算才并入摘要、游标前进。
持久化（写回会话）在调用方（chat_service）；本模块只管装配与游标推进。

票 21 的**工具结果清理**同样落在这里：`trim_tool_result` 把"已被引用"的来源收缩为
元信息（纯函数，只丢全文、不丢可回溯）。**什么时候收**由代理循环（agent.py）决定，
枚举/编号查询的豁免由调用方按既有判据（chat_service 的 `compress_exempt`）传给循环。

以下**不在**本模块范围：真实分词器（票 19）、摘要的持久化与豁免判据（调用方）。
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Collection, Sequence

from app.core.llm import LLM
from app.core.prompt import format_turn
from app.utils.text import approx_token_count, truncate

logger = logging.getLogger(__name__)



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
        return approx_token_count(text)


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


def plan_exempt(history: Sequence[dict] | None, *, previous: str | None = None,
                previous_upto: str | None = None) -> ContextPlan:
    """本问豁免压缩（票 20）：历史原样透传，已有摘要照带 —— 但**按游标切掉已被摘要覆盖的轮次**。

    为什么不整段塞进去：游标通常落在加载窗口内，不切的话同一轮会**既在摘要里、又在原文里**
    出现两次（非豁免路径靠 `assemble_context` 的尾段切片避开了这件事，这里必须同样避开）。
    窗口之外的更早对话只存在于摘要里，所以摘要必须照带 —— 丢掉它才是真的漏上下文。
    """
    hist = list(history or [])
    if not previous:
        return ContextPlan(summary=None, kept=hist)
    return ContextPlan(summary=previous, kept=hist[_covered_turns(hist, previous_upto):],
                       cursor=previous_upto or None)


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


# ---------- 工具结果清理（票 21）----------

TOOL_HEAD_CHARS = 120      # 收缩后每条来源保留的首部片段长度

_CITE_RE = re.compile(r"\[来源[:：]\s*([^\]\n]+)\]")
_CITE_DOC_PAGE = re.compile(r"^(.*?)[,，]\s*第?\s*(\d+)\s*页\s*$")


def cited_chunk_ids(text: str, sources: Sequence[dict] | None) -> set[str]:
    """模型这次回复里点名引用过的来源（票 21）。

    判据只用**可机读的锚点**：chunk_id 原样出现，或 `[来源: 文档名, 第X页]` 里文档名与页码同时命中。
    不做模糊匹配 —— 宁可晚一轮再收缩，也不误收缩模型还要用的内容。
    """
    if not text:
        return set()
    marked: set[tuple[str, int]] = set()
    for m in _CITE_RE.finditer(text):
        doc_page = _CITE_DOC_PAGE.match(m.group(1).strip())
        if doc_page:
            marked.add((doc_page.group(1).strip(), int(doc_page.group(2))))

    out: set[str] = set()
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        cid = s.get("chunk_id")
        if not cid:
            continue
        if cid in text:
            out.add(cid)
            continue
        key = (str(s.get("doc_name") or "").strip(), int(s.get("page") or 0))
        if key[0] and key in marked:
            out.add(cid)
    return out


def trim_tool_result(result: dict, cited: Collection[str], *,
                     keep_head: int = TOOL_HEAD_CHARS) -> dict | None:
    """把**已被引用**的来源收缩为元信息；没有可收缩的就返回 None（调用方保持原样，不做无谓替换）。

    只丢"全文"：chunk_id / doc_id / doc_name / page 一个不少，text 留首部片段 ——
    引用仍能回溯到原文片段，代理也还知道这块讲过什么。
    结果里没被引用的块**原样保留**：不提前收缩模型还要用的内容。
    `trimmed` 是个布尔标记（不是散文）—— 让模型知道这段是首部而非全块，代价一个键。
    """
    sources = result.get("sources")
    if not isinstance(sources, list) or not sources:
        return None
    shrunk = False
    kept: list = []
    for s in sources:
        if isinstance(s, dict) and s.get("chunk_id") in cited and len(str(s.get("text") or "")) > keep_head:
            s = dict(s)
            s["text"] = truncate(s["text"], keep_head)
            shrunk = True
        kept.append(s)
    if not shrunk:
        return None
    out = dict(result)
    out["sources"] = kept
    out["trimmed"] = True
    return out
