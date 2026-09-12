"""滚动摘要落库 + 增量复用（票 18 / #25）。

摘要写在**会话上**，长对话不因刷新页面 / 重启服务回到起点；未摘要的尾部没超预算就复用。
缝：注入的 token 计数器与摘要器（stub 下确定性、无网络、无真实 LLM）。
"""
from __future__ import annotations

from app.services import chat_service
from tests.helpers import register_and_kb


class CountingSummarizer:
    """数调用次数的摘要器 —— 「增量复用」看的就是这个数。"""

    def __init__(self, out: str = "（更早的对话摘要）"):
        self.out = out
        self.calls: list = []

    def summarize(self, messages):
        self.calls.append(list(messages))
        return self.out


class TurnCounter:
    """按「几轮」计数 —— 预算判定不依赖问答文本有多长，测试也就不会随文案变动而脆断。"""

    def count(self, text: str) -> int:
        return (text or "").count("用户：")


def _setup(client, name: str):
    """注册 + 建库 + 装上可控的计数器与摘要器，返回 (user, kb, db, rt, summarizer)。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = get_runtime()
    summarizer = CountingSummarizer()
    rt.token_counter = TurnCounter()
    rt.context_summarizer_factory = lambda llm: summarizer
    return user, kb, db, rt, summarizer


def test_the_summary_rolls_only_when_the_tail_overflows_then_is_reused(client, monkeypatch):
    """尾部超了才滚一次（且只滚未覆盖的那几轮）；不超就复用，摘要器一次都不调。"""
    from app.models.entities import ChatSession

    user, kb, db, rt, s = _setup(client, "sum_persist")
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 3)   # 装下 3 轮
    try:
        chat_service.answer(db, rt, user, kb, "第一个问题", "sum-session-1")

        for i in (2, 3, 4):      # 到第 4 问时历史 3 轮：正好装得下
            chat_service.answer(db, rt, user, kb, "第%d个问题" % i, "sum-session-1")
        assert s.calls == [], "3 轮以内装得下，不该压"

        chat_service.answer(db, rt, user, kb, "第五个问题", "sum-session-1")

        sess = db.get(ChatSession, "sum-session-1")
        assert len(s.calls) == 1, "4 轮超预算，滚一次"
        assert sess.summary == s.out
        assert sess.summary_upto, "游标要落在被摘要的最后一轮上"
        cursor, calls = sess.summary_upto, len(s.calls)

        for i in (6, 7):        # 尾部又回到预算内 -> 复用
            chat_service.answer(db, rt, user, kb, "第%d个问题" % i, "sum-session-1")
        assert len(s.calls) == calls, "尾部没超预算就该复用，不许再压一遍"
        assert db.get(ChatSession, "sum-session-1").summary_upto == cursor

        chat_service.answer(db, rt, user, kb, "第八个问题", "sum-session-1")
        assert len(s.calls) == calls + 1, "尾部再次超预算 -> 再滚一次"
        # 增量：第二次滚动只喂**未覆盖的尾部**+旧摘要，不再把最早那几轮重复喂一遍
        fed = " ".join(str(m) for m in s.calls[-1])
        assert "第一个问题" not in fed and s.out in fed
    finally:
        db.close()


def test_a_reloaded_session_puts_the_persisted_summary_back_in_the_prompt(client, monkeypatch):
    """「重启后不回到起点」：摘要只在库里（进程内没有任何状态），仍要装进提示词。"""
    from app.models.entities import ChatMessage, ChatSession

    user, kb, db, rt, s = _setup(client, "sum_reload")
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 40)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        # 模拟「上一次进程」留下的东西：两轮对话 + 一段覆盖到第一轮的摘要
        sess = ChatSession(id="sum-session-2", user_id=user.id, kb_id=kb, title="")
        db.add(sess)
        db.commit()
        ids = []
        for i in (1, 2):
            for role, text in (("user", "早先问题%d" % i), ("assistant", "早先回答%d" % i)):
                m = ChatMessage(session_id=sess.id, role=role, content=text)
                db.add(m)
                db.commit()
                db.refresh(m)
                ids.append(m.id)
        sess.summary = "（上次留在库里的摘要）"
        sess.summary_upto = ids[1]              # 覆盖到第一轮的最后一条消息
        db.commit()

        prep = chat_service.prepare(db, rt, user, kb, "接着问", "sum-session-2")

        assert s.calls == [], "尾部就一轮，装得下 —— 不该重新摘要"
        assert "（上次留在库里的摘要）" in prep.prompt
    finally:
        db.close()


def test_compression_off_writes_no_summary(client, monkeypatch):
    """关掉压缩：不摘要、不落库（开关语义与今天一致）。"""
    from app.models.entities import ChatSession

    user, kb, db, rt, s = _setup(client, "sum_off")
    monkeypatch.setattr(chat_service.get_settings(), "context_compress", False)
    try:
        for q in ("问题一", "问题二", "问题三"):
            chat_service.answer(db, rt, user, kb, q, "sum-session-3")

        assert s.calls == []
        assert (db.get(ChatSession, "sum-session-3").summary or "") == ""
    finally:
        db.close()


def test_an_existing_db_gains_the_new_columns(tmp_path):
    """老库缺新列时补上 —— 不然升级后第一句话就是 500（本仓库没有迁移框架）。"""
    from sqlalchemy import create_engine, inspect, text

    from app.db.migrate import ensure_sqlite_columns

    engine = create_engine("sqlite:///%s" % (tmp_path / "old.db").as_posix())
    with engine.begin() as conn:      # 造一张「老版本」的会话表：没有 summary 两列
        conn.execute(text("CREATE TABLE chat_sessions ("
                          "id VARCHAR(32) PRIMARY KEY, user_id VARCHAR(32), "
                          "kb_id VARCHAR(32), title VARCHAR(256))"))

    added = ensure_sqlite_columns(engine)

    have = {c["name"] for c in inspect(engine).get_columns("chat_sessions")}
    assert {"summary", "summary_upto"} <= have
    assert set(added) == {"chat_sessions.summary", "chat_sessions.summary_upto"}


class LabeledCounter:
    """带口径的计数器 —— 只有 `label` 非空，token 数字才允许对外报（票 19）。"""

    label = "Qwen/test"

    def count(self, text: str) -> int:
        return len(text or "")


def test_the_trace_reports_tokens_only_with_a_real_tokenizer(client, monkeypatch):
    """有真实分词器：报压缩前/后 token 与口径；没有：只报「不可用」，不拿估算顶替。"""
    user, kb, db, rt, s = _setup(client, "sum_tokens")
    rt.token_counter = LabeledCounter()           # 真实分词器的口径（label 非空）
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 3)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        for i in range(1, 6):
            out = chat_service.answer(db, rt, user, kb, "第%d个问题" % i, "sum-session-tokens")
        usage = out["trace"]["context_tokens"]
        assert usage["tokenizer"] == "Qwen/test"
        assert usage["tokens_before"] > usage["tokens_after"]     # 确实省了

        rt.token_counter = TurnCounter()          # 换回无口径的计数器
        out2 = chat_service.answer(db, rt, user, kb, "接着问", "sum-session-tokens")
        usage2 = out2["trace"]["context_tokens"]
        assert usage2["tokens_before"] is None and usage2["tokenizer"] == ""
    finally:
        db.close()
