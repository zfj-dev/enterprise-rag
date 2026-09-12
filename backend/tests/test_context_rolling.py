"""滚动摘要的纯逻辑（票 18 / #25）：已有摘要覆盖的部分不再重算，只滚动未摘要的尾部。

缝：注入的 token 计数器 + 摘要器；stub 下确定性、无网络、无真实 LLM。
"""
from __future__ import annotations

from app.core.context import TokenCounter, assemble_context


class StubCounter(TokenCounter):
    """每字符 1 token，便于精确构造「超 / 不超」。"""

    def count(self, text: str) -> int:
        return len(text or "")


class RecordingSummarizer:
    def __init__(self, out: str = "（累计摘要）"):
        self.out = out
        self.calls: list = []

    def summarize(self, messages):
        self.calls.append(list(messages))
        return self.out


def _hist(n: int, start: int = 1) -> list[dict]:
    """带消息 id 的历史 —— 游标按「最后一条已摘要消息的 id」记。"""
    return [{"user": "问题%d" % i, "assistant": "回答%d" % i, "ids": ["u%d" % i, "a%d" % i]}
            for i in range(start, start + n)]


def _plan(history, **kw):
    kw.setdefault("budget", 10_000)
    kw.setdefault("keep_recent", 1)
    return assemble_context(history, count_tokens=StubCounter().count, **kw)


def test_a_covering_summary_is_reused_without_calling_the_summarizer():
    """已有摘要覆盖了前 3 轮、尾部 2 轮装得下 -> 直接复用，**一次都不调摘要器**。"""
    h = _hist(5)
    s = RecordingSummarizer()
    plan = _plan(h, summarize=s.summarize, previous="（早先聊过 1-3）", previous_upto="a3")

    assert s.calls == []
    assert plan.summary == "（早先聊过 1-3）"
    assert plan.kept == h[3:]                  # 只带未摘要的尾部
    assert plan.cursor == "a3"                 # 游标不动
    assert plan.dropped == 0


def test_the_unsummarized_tail_is_rolled_in_only_when_it_overflows():
    """尾部再次超预算 -> 才滚一次：摘要器收到尾部更早那几轮，游标前进。"""
    h = _hist(5)
    s = RecordingSummarizer("（累计摘要 1-4）")
    plan = _plan(h, budget=1, keep_recent=1, summarize=s.summarize,
                 previous="（早先聊过 1-3）", previous_upto="a3")

    assert len(s.calls) == 1
    assert plan.summary == "（累计摘要 1-4）"
    assert plan.kept == h[4:]                  # 只保留最近一轮原文
    assert plan.cursor == "a4"                 # 游标前进到刚被摘要的最后一轮
    assert plan.dropped == 1


def test_the_previous_summary_is_fed_back_so_it_accumulates():
    """新一轮摘要必须看到旧摘要 —— 否则每滚一次就把更早的对话丢掉一遍。"""
    h = _hist(5)
    s = RecordingSummarizer()
    _plan(h, budget=1, keep_recent=1, summarize=s.summarize,
          previous="（早先聊过 1-3）", previous_upto="a3")

    assert s.calls[0][0]["assistant"] == "（早先聊过 1-3）"     # 旧摘要摆在摘要器看到的最前面


def test_a_cursor_that_fell_out_of_the_window_covers_nothing():
    """游标指向的历史已经不在窗口里 -> 窗口内全是未摘要的，按尾部处理（不丢）。"""
    h = _hist(3, start=10)
    s = RecordingSummarizer()
    plan = _plan(h, budget=1, keep_recent=1, summarize=s.summarize,
                 previous="（很久以前）", previous_upto="a3")

    assert len(s.calls) == 1
    assert plan.dropped == 2
    assert plan.cursor == "a11"


def test_history_without_ids_still_works():
    """没带 id 的历史（旧调用方 / 测试 stub）不该炸 —— 只是游标推不动。"""
    h = [{"user": "问", "assistant": "答"} for _ in range(3)]
    s = RecordingSummarizer()
    plan = _plan(h, budget=1, keep_recent=1, summarize=s.summarize)

    assert plan.summary == "（累计摘要）"
    assert plan.cursor is None
