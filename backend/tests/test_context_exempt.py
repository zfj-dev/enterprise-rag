"""枚举意图豁免压缩（票 20 / #27）：刻意注入大批块的那类问题，一点都不许压。

回归靶子：长会话里问「列出所有表格」，历史被压成摘要 + 注入块被截断 -> 列不全。
"""
from __future__ import annotations

from app.services import chat_service
from tests.helpers import register_and_kb, wait_until

TABLE_DOC = ("表 3.1 实验环境配置\n\n"
             "| 项 | 值 |\n| --- | --- |\n| 操作系统 | Windows 11 |\n| CUDA | 12.3 |\n")


class CountingSummarizer:
    def __init__(self, out: str = "（更早的对话摘要）"):
        self.out = out
        self.calls: list = []

    def summarize(self, messages):
        self.calls.append(list(messages))
        return self.out


class TurnCounter:
    """按「几轮」计数，便于精确构造超预算。"""

    def count(self, text: str) -> int:
        return (text or "").count("用户：")


def _setup(client, name: str):
    """建库 + 传一份带表格的文档 + 装上可控的计数器/摘要器。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    up = client.post("/api/v1/documents?kb_id=%s" % kb, headers=H,
                     files={"file": ("tables.txt", TABLE_DOC, "text/plain")}).json()
    assert wait_until(lambda: client.get("/api/v1/documents/%s" % up["id"], headers=H)
                      .json().get("status") in ("indexed", "failed")), "文档未入库"

    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = get_runtime()
    summarizer = CountingSummarizer()
    rt.token_counter = TurnCounter()
    rt.context_summarizer_factory = lambda llm: summarizer
    return user, kb, db, rt, summarizer


def _long_session(user, kb, db, rt, session_id: str, rounds: int = 4) -> None:
    """先把这个会话聊长 —— 长到非枚举问题一定会触发压缩。"""
    for i in range(rounds):
        chat_service.answer(db, rt, user, kb, "第%d个问题" % i, session_id)


def test_an_enumeration_question_is_never_compressed(client, monkeypatch):
    """枚举问题：不新压、游标不动；已有摘要照带但**不重复**窗口内的原文。"""
    from app.models.entities import ChatSession

    user, kb, db, rt, s = _setup(client, "exempt_enum")
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 1)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        _long_session(user, kb, db, rt, "exempt-1")
        before = len(s.calls)
        sess = db.get(ChatSession, "exempt-1")
        summary, cursor = sess.summary, sess.summary_upto
        assert summary, "先聊到压出摘要，这条用例才有意义"

        prep = chat_service.prepare(db, rt, user, kb, "列出所有表格", "exempt-1")

        assert len(s.calls) == before, "枚举问题不该触发压缩"
        assert prep.trace["compress_exempt"] is True
        assert summary in prep.prompt          # 窗口外的更早对话（摘要）照带
        assert "第0个问题" not in prep.prompt   # 但已被摘要覆盖的原文不许再来一遍
        assert "Windows 11" in prep.prompt     # 注入的表格块在

        db.refresh(sess)
        assert (sess.summary, sess.summary_upto) == (summary, cursor), "豁免不推进摘要与游标"
    finally:
        db.close()


def test_the_exemption_does_not_depend_on_the_budget(client, monkeypatch):
    """预算再小也豁免：压与不压不取决于省多少 token。"""
    user, kb, db, rt, s = _setup(client, "exempt_budget")
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 0)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        _long_session(user, kb, db, rt, "exempt-2")
        before = len(s.calls)

        prep = chat_service.prepare(db, rt, user, kb, "有哪些图片", "exempt-2")

        assert len(s.calls) == before and prep.trace["compress_exempt"] is True
    finally:
        db.close()


def test_a_named_reference_question_is_exempt_too(client, monkeypatch):
    """具体编号（表3.1）与枚举走同一条豁免 —— 复用既有意图判据，不另起一套。"""
    user, kb, db, rt, s = _setup(client, "exempt_ref")
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 1)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        _long_session(user, kb, db, rt, "exempt-3")
        before = len(s.calls)

        prep = chat_service.prepare(db, rt, user, kb, "表3.1的内容是什么", "exempt-3")

        assert len(s.calls) == before and prep.trace["compress_exempt"] is True
    finally:
        db.close()


def test_a_normal_question_still_compresses(client, monkeypatch):
    """豁免不外溢：普通问题照压不误。"""
    user, kb, db, rt, s = _setup(client, "exempt_normal")
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 1)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        _long_session(user, kb, db, rt, "exempt-4")
        before = len(s.calls)

        prep = chat_service.prepare(db, rt, user, kb, "这个文档讲了什么", "exempt-4")

        assert len(s.calls) == before + 1, "普通问题该压还得压"
        assert prep.trace["compress_exempt"] is False
    finally:
        db.close()


def test_a_summary_is_not_duplicated_by_the_exempt_path():
    """纯函数侧锁住同一条：切掉被摘要覆盖的轮次，旧摘要照带。"""
    from app.core.context import plan_exempt

    hist = [{"user": "问1", "assistant": "答1", "ids": ["u1", "a1"]},
            {"user": "问2", "assistant": "答2", "ids": ["u2", "a2"]}]

    plan = plan_exempt(hist, previous="（早先聊过 1）", previous_upto="a1")

    assert plan.summary == "（早先聊过 1）"
    assert plan.kept == hist[1:]          # 被摘要覆盖的第一轮不再重复出现
    assert plan.cursor == "a1"
    assert plan_exempt(hist).summary is None      # 没有摘要时就是纯透传


class LabeledCounter:
    """带口径的计数器（label 非空 = 真实分词器在）。"""

    label = "Qwen/test"

    def count(self, text: str) -> int:
        return len(text or "")


def test_an_exempt_round_reports_no_reduction_instead_of_a_fake_number(client, monkeypatch):
    """豁免轮没有「降了多少」可报：报 0% 甚至负值都是误导（票 20 修）。"""
    user, kb, db, rt, s = _setup(client, "exempt_tokens")
    rt.token_counter = LabeledCounter()
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 1)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        _long_session(user, kb, db, rt, "exempt-5")
        prep = chat_service.prepare(db, rt, user, kb, "列出所有表格", "exempt-5")

        usage = prep.trace["context_tokens"]
        assert usage["tokens_before"] is None and usage["tokens_after"] is None
        # 豁免的原因走 **exempt_note**：「没有真实分词器」是另一回事，两个挤一个字段会让
        # 报告把「本问豁免压缩」当成「没接分词器」的原因（#53）。note 这里必须是空的。
        assert "豁免压缩" in usage["exempt_note"]
        assert usage["note"] == ""
    finally:
        db.close()
