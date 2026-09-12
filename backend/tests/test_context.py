"""上下文预算装配（票 17）：未超预算原样透传，超了才压缩并保留最近若干轮。

缝：注入的 token 计数器 + 摘要器；stub 下确定性、无网络、无真实 LLM。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.core.context import ApproxTokenCounter, TokenCounter, assemble_context


class StubCounter(TokenCounter):
    """每字符 1 token，便于精确构造"超 / 不超"。"""

    def count(self, text: str) -> int:
        return len(text or "")


class RecordingSummarizer:
    def __init__(self, out: str = "（早先聊过）"):
        self.out = out
        self.calls: list = []

    def summarize(self, messages):
        self.calls.append(list(messages))
        return self.out


def _hist(n: int) -> list[dict]:
    return [{"user": f"问题{i}", "assistant": f"回答{i}"} for i in range(n)]


# ---------- 纯函数 ----------

def test_under_budget_passes_through_untouched():
    """未超预算：原样透传，摘要器一次都不被调用（零额外 LLM 调用）。"""
    h = _hist(3)
    s = RecordingSummarizer()
    plan = assemble_context(h, budget=10_000, keep_recent=1,
                            count_tokens=StubCounter().count, summarize=s.summarize)
    assert plan.kept == h
    assert plan.summary is None
    assert plan.dropped == 0
    assert s.calls == []


def test_over_budget_compresses_and_keeps_recent():
    """超预算：更早的对话被摘要，最近若干轮原文保留。"""
    h = _hist(5)
    s = RecordingSummarizer("（早先聊过 1、2）")
    plan = assemble_context(h, budget=30, keep_recent=2,
                            count_tokens=StubCounter().count, summarize=s.summarize)
    assert plan.kept == h[-2:]
    assert plan.dropped == 3
    assert plan.summary == "（早先聊过 1、2）"
    assert len(s.calls) == 1
    assert s.calls[0] == h[:3]


def test_nothing_older_than_keep_recent_does_not_summarize():
    """历史轮数不超过 keep_recent 时没有"更早的"可压 —— 不该白调一次 LLM。"""
    h = _hist(2)
    s = RecordingSummarizer()
    plan = assemble_context(h, budget=1, keep_recent=3,
                            count_tokens=StubCounter().count, summarize=s.summarize)
    assert plan.kept == h
    assert plan.summary is None
    assert s.calls == []


def test_empty_history():
    s = RecordingSummarizer()
    plan = assemble_context([], budget=1, keep_recent=3,
                            count_tokens=StubCounter().count, summarize=s.summarize)
    assert plan.kept == [] and plan.summary is None and s.calls == []


def test_approx_counter_counts_cjk_and_ascii_words():
    c = ApproxTokenCounter()
    assert c.count("") == 0
    assert c.count("比亚迪") == 3           # 3 个汉字
    assert c.count("hello world") == 2      # 2 个 ASCII 词
    assert c.count("比亚迪 bge") == 4       # 3 汉字 + 1 词
    assert c.count("比亚迪") == c.count("比亚迪")   # 确定性


# ---------- 接进问答链路 ----------

def _session():
    from app.db.session import SessionLocal
    return SessionLocal()


def _seed_history(client, name: str, turns: int):
    """注册用户 + 建库 + 造一段多轮历史（created_at 递增，保证顺序确定）。"""
    from app.db.session import SessionLocal
    from app.models.entities import ChatMessage, ChatSession, KnowledgeBase, User

    tok = client.post("/api/v1/auth/register",
                      json={"username": name, "password": "pw123456"}).json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == name).first()
        kb = KnowledgeBase(owner_id=u.id, name=f"{name}-kb", description="")
        db.add(kb)
        db.commit()
        db.refresh(kb)
        sess = ChatSession(user_id=u.id, kb_id=kb.id, title="t")
        db.add(sess)
        db.commit()
        db.refresh(sess)
        t0 = datetime(2026, 1, 1)
        for i in range(turns):
            db.add(ChatMessage(session_id=sess.id, role="user", content=f"问题{i}" * 20,
                               created_at=t0 + timedelta(seconds=2 * i)))
            db.add(ChatMessage(session_id=sess.id, role="assistant", content=f"回答{i}" * 20,
                               created_at=t0 + timedelta(seconds=2 * i + 1)))
        db.commit()
        return H, kb.id, sess.id, u.id
    finally:
        db.close()


def _prepare_with(client, monkeypatch, name: str, budget: int, compress: bool = True):
    """造历史 → 装上带 stub 的 Runtime → 调 prepare，返回 (prep, summarizer)。"""
    import app.api.deps as deps
    import app.config as cfg
    from app.core.container import build_runtime
    from app.models.entities import User
    from app.services import chat_service

    H, kb_id, sess_id, uid = _seed_history(client, name, turns=10)
    summ = RecordingSummarizer()
    rt = build_runtime()
    rt.token_counter = StubCounter()
    rt.context_summarizer_factory = lambda llm: summ
    deps._runtime = rt
    monkeypatch.setattr(cfg.get_settings(), "context_token_budget", budget)
    monkeypatch.setattr(cfg.get_settings(), "context_compress", compress)

    db = _session()
    try:
        user = db.query(User).filter(User.id == uid).first()
        prep = chat_service.prepare(db, rt, user, kb_id, "继续", session_id=sess_id)
    finally:
        db.close()
    return prep, summ


def test_prepare_injects_summary_when_over_budget(client, monkeypatch):
    """超预算时：更早的对话进【更早对话摘要】，最近若干轮仍以原文保留。"""
    prep, summ = _prepare_with(client, monkeypatch, "ctxuser", budget=200)
    assert "【更早对话摘要】" in prep.prompt
    assert "（早先聊过）" in prep.prompt
    assert prep.trace.get("context_dropped", 0) > 0
    assert len(summ.calls) == 1


def test_prepare_no_summary_when_under_budget(client, monkeypatch):
    """预算充足时历史原样进 prompt，不出现摘要块，且不额外调用 LLM。"""
    prep, summ = _prepare_with(client, monkeypatch, "ctxbig", budget=10_000_000)
    assert "【更早对话摘要】" not in prep.prompt
    assert prep.trace.get("context_dropped", 0) == 0
    assert summ.calls == []


# ---------- 降级与开关（审查发现） ----------

def test_summarizer_failure_degrades_to_passthrough():
    """摘要器挂掉（如网络/配额）时退回原样，而不是让整个回答崩掉。"""
    class Boom:
        def summarize(self, messages):
            raise RuntimeError("node down")

    h = _hist(5)
    plan = assemble_context(h, budget=1, keep_recent=2,
                            count_tokens=StubCounter().count, summarize=Boom().summarize)
    assert plan.kept == h
    assert plan.summary is None
    assert plan.dropped == 0


def test_empty_summary_does_not_drop_history():
    """摘要返回空串时，宁可原样保留，也不静默丢掉更早的对话。"""
    h = _hist(5)
    s = RecordingSummarizer("   ")
    plan = assemble_context(h, budget=1, keep_recent=2,
                            count_tokens=StubCounter().count, summarize=s.summarize)
    assert plan.kept == h
    assert plan.summary is None
    assert plan.dropped == 0
    assert len(s.calls) == 1     # 试过了，只是结果不可用


def test_off_switch_disables_compression(client, monkeypatch):
    """关闭压缩开关：即使远超预算也不压缩、不调摘要器。"""
    prep, summ = _prepare_with(client, monkeypatch, "ctxoff", budget=200, compress=False)
    assert "【更早对话摘要】" not in prep.prompt
    assert prep.trace.get("context_dropped", 0) == 0
    assert summ.calls == []


# ---------- 历史装配 ----------

def test_load_history_pairs_turns_written_in_the_same_second(client):
    """同秒写入的一轮 user/assistant 也必须配对正确。

    回归：created_at 秒级精度时同秒全相等，而 sqlite 对相等的排序键 ASC/DESC 都按 rowid
    返回（DESC 并不翻转），_load_history 再 reversed() 就把整段历史错开一格 —— 配成
    {问2,答1}、{问1,答0} 这种张冠李戴。这里刻意不给 created_at，全部走 ORM 默认值。
    """
    from app.db.session import SessionLocal
    from app.models.entities import ChatMessage, ChatSession
    from app.services.chat_service import _load_history
    from tests.helpers import register_and_kb

    _, uid, kb_id = register_and_kb(client, "histtie")
    db = SessionLocal()
    try:
        sess = ChatSession(user_id=uid, kb_id=kb_id, title="t")
        db.add(sess)
        db.commit()
        db.refresh(sess)
        for i in range(3):
            db.add(ChatMessage(session_id=sess.id, role="user", content=f"问{i}"))
            db.add(ChatMessage(session_id=sess.id, role="assistant", content=f"答{i}"))
        db.commit()

        got = _load_history(db, sess.id)
        assert [(t["user"], t["assistant"]) for t in got] == [
            ("问0", "答0"), ("问1", "答1"), ("问2", "答2")]
        assert all(len(t["ids"]) == 2 for t in got)      # 每轮带上它那两条消息的 id（票 18 的游标）
    finally:
        db.close()
