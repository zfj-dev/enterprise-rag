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


def test_no_compression_is_not_reported_as_a_zero_reduction(client, monkeypatch):
    """历史没超预算 = 这一问**压根没压**。token 记 None（不进降幅）并写明原因。

    报成 0% 等于把「没发生」写成「发生了但结果是 0」；口径仍要留在原地 ——
    这不是「没接分词器」（#57）。
    """
    user, kb, db, rt, _ = _setup(client, "sum_nocompress")
    rt.token_counter = LabeledCounter()
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 100000)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        for i in range(1, 4):
            out = chat_service.answer(db, rt, user, kb, "第%d个问题" % i, "sum-nocompress")
        usage = out["trace"]["context_tokens"]

        assert usage["tokens_before"] is None and usage["tokens_after"] is None
        assert usage["tokenizer"] == "Qwen/test"          # 口径还在：不是「没分词器」
        assert usage["note"] == ""                        # 更不能错记成「拿不到分词器」
        assert "未超预算" in usage["no_compress_note"]
    finally:
        db.close()


def test_answer_carries_the_fields_the_eval_core_reads(client):
    """非流式入口也要带 `context` / `citation_coverage`。

    评测走的正是这条路（RGB 段）：只手挑 answer + sources 会把 token 口径与
    引用覆盖率整段丢掉，报告只能写「不可用」并错怪分词器（#58）。
    """
    user, kb, db, rt, _ = _setup(client, "sum_fields")
    try:
        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")

        assert out["context"] is not None and "citation_coverage" in out
    finally:
        db.close()


class GroundlessLLM:
    """假装是真模型：生成一个**在来源里找不到依据**的答案，逐句校验回「一条都不支撑」。

    演示用的 FakeLLM 会被 `_finish` 跳过（不调校验），所以要 `is_fake = False` 才走得到守门。
    """

    is_fake = False

    def stream(self, messages, usage=None):
        prompt = messages[-1]["content"]
        if "是否被参考资料支撑" in prompt:
            yield '{"claims":[{"claim":"答案是 42。","supported":false}]}'
        else:
            yield "答案是 42。"


def test_a_groundless_answer_is_turned_into_a_refusal(client, monkeypatch):
    """接线测试（票 B）：一条依据都没有 -> 落库与返回都改成拒答，trace 记下拦过。

    这条走的是**真实的 chat_service 链路**（不是手写一个 verification dict）——
    守门接没接上，只有从 `answer()` 一路问出来才算数。
    """
    user, kb, db, rt, _ = _setup(client, "sum_guard")
    rt.llm = GroundlessLLM()
    monkeypatch.setattr(chat_service, "retrieve_candidates",
                        lambda *a, **k: [{"chunk_id": "c1", "content": "无关内容",
                                          "page_num": 1, "score": 0.9}])
    try:
        out = chat_service.answer(db, rt, user, kb, "营收多少", "sum-guard")

        assert "无法确定" in out["answer"] and "42" not in out["answer"]
        assert out["trace"]["citation_guard"]["fired"] is True
        assert out["trace"]["citation_coverage"] == 0.0
    finally:
        db.close()


def test_the_no_source_refusal_keeps_its_own_wording(client, monkeypatch):
    """一条来源都没有时，别把上游「未检索到可引用的内容」那句换成「论断没依据」。

    两句都是拒答，但说的是**不同的**原因（没检索到 / 检索到了但撑不住）——
    覆盖掉就是把真原因说没了（#62 两轴审查抓到）。
    """
    user, kb, db, rt, _ = _setup(client, "sum_nosrc")
    rt.llm = GroundlessLLM()
    monkeypatch.setattr(chat_service, "retrieve_candidates", lambda *a, **k: [])
    try:
        out = chat_service.answer(db, rt, user, kb, "营收多少", "sum-nosrc")

        assert "未检索到可引用的内容" in out["answer"]
        assert out["trace"]["citation_guard"]["fired"] is False
    finally:
        db.close()


class BoomSummarizer:
    """摘要器炸了（网络/配额）—— 退回不压缩，但**原因不是**「历史没超预算」。"""

    def summarize(self, messages):
        raise RuntimeError("网络挂了")


def test_a_failed_summarizer_is_not_blamed_on_the_budget(client, monkeypatch):
    """摘要失败 -> 报告必须说「摘要失败」。

    从「没有摘要」反推成「未超预算」就是编了一个原因 —— 而这条链路真会报出
    「历史 N tokens 未超预算 M」且 N > M 的自相矛盾句子（#57 的两轴审查抓到）。
    """
    user, kb, db, rt, _ = _setup(client, "sum_boom")
    rt.token_counter = LabeledCounter()
    rt.context_summarizer_factory = lambda llm: BoomSummarizer()
    monkeypatch.setattr(chat_service.get_settings(), "context_token_budget", 3)
    monkeypatch.setattr(chat_service.get_settings(), "context_keep_recent", 1)
    try:
        for i in range(1, 6):
            out = chat_service.answer(db, rt, user, kb, "第%d个问题" % i, "sum-boom")
        note = out["trace"]["context_tokens"]["no_compress_note"]

        assert "摘要失败" in note
        assert "未超预算" not in note          # 别把失败说成「本来就不需要压」
        assert out["trace"]["context_tokens"]["tokenizer"] == "Qwen/test"
        assert out["trace"]["context_tokens"]["tokens_before"] is None
    finally:
        db.close()
