"""费用折算 + 价格表（票 28 / #35）：token 按模型单价折成费用，未知单价**不按 0 算**。

价格表从外部注入 —— 确定性、无网络；覆盖项从配置里读，配错了也不该让服务起不来。
"""
from __future__ import annotations

from app.core.pricing import Price, PriceTable, cost_of
from tests.helpers import register_and_kb

KNOWN = {"model": "m1", "input_tokens": 2000, "output_tokens": 1000,
         "source": "provider", "source_note": "账单口径：provider 返回的 usage"}
TABLE = PriceTable({"m1": {"input": 0.001, "output": 0.002}})     # 元 / 1K token


# ---------- 折算 ----------

def test_a_known_model_is_converted_with_its_unit_prices():
    got = cost_of(KNOWN, TABLE)

    # 2K 输入 × 0.001 + 1K 输出 × 0.002 = 0.004
    assert got["cost"] == 0.004
    assert "m1" in got["price_note"] and "1K token" in got["price_note"]   # 口径跟着数字


def test_an_unknown_model_is_labelled_not_priced_at_zero():
    got = cost_of({**KNOWN, "model": "某个没配过的模型"}, TABLE)

    assert got["cost"] is None                      # **不是 0** —— 0 元看起来像「没花钱」
    assert "单价未知" in got["price_note"] and "某个没配过的模型" in got["price_note"]
    assert "LLM_PRICE_OVERRIDES" in got["price_note"]      # 告诉人怎么补上


def test_unavailable_tokens_make_the_cost_unavailable_too():
    got = cost_of({**KNOWN, "input_tokens": None, "output_tokens": None,
                   "source": "unavailable", "source_note": "不可用：provider 未返回 usage"},
                  TABLE)

    assert got["cost"] is None
    assert "token 不可用" in got["price_note"]


def test_the_configuration_overrides_the_builtin_price():
    """覆盖价优先于内置价；覆盖还能**新增**内置表里没有的模型。"""
    table = PriceTable({"fake": {"input": 9.0, "output": 9.0}, "m9": {"input": 1.0, "output": 2.0}})
    got = cost_of({"model": "fake", "input_tokens": 1000, "output_tokens": 0}, table)

    assert got["cost"] == 9.0                       # 用覆盖价，不用内置的 0
    assert table.price_of("m9") is not None         # 覆盖项可以补上表里没有的模型


def test_no_real_price_is_built_in():
    """真实模型的单价会随供应商调价失真 —— 本项目**不内置一份离线核实不了的价目表**。

    宁可先报「单价未知」让人去配，也不给一个看着正常的错数字。
    """
    table = PriceTable()

    for model in ("deepseek-chat", "qwen-plus", "qwen-turbo", "gpt-4o-mini"):
        assert table.price_of(model) is None, "%s 不该有内置价" % model
        got = cost_of({"model": model, "input_tokens": 100, "output_tokens": 100}, table)
        assert got["cost"] is None and "单价未知" in got["price_note"]


def test_the_price_note_says_where_the_price_came_from_and_the_token_caliber():
    """费用数字要自洽可读：单价从哪儿来、token 是账单还是估算，都写在口径里。"""
    got = cost_of(KNOWN, TABLE)                     # TABLE 是配置覆盖价，token 是账单口径

    assert "配置覆盖价" in got["price_note"]
    assert "账单口径" in got["price_note"]
    assert "内置价" not in got["price_note"]


def test_the_builtin_table_prices_the_local_fake_model_at_zero():
    """fake 是**构造上**的 0（本地假模型不发请求）—— 这与「未知按 0」是两回事。"""
    got = cost_of({"model": "fake", "input_tokens": 10, "output_tokens": 5}, PriceTable())

    assert got["cost"] == 0.0
    assert "内置价" in got["price_note"]


# ---------- 配置解析（配错不该让服务起不来，但要留下问题清单） ----------

def test_a_broken_override_blob_is_ignored_instead_of_raising():
    table = PriceTable("{这不是 JSON")

    assert table.price_of("fake") is not None            # 内置价照旧可用
    assert table.warnings                                 # 但问题必须留下痕迹，不能静默


def test_broken_overrides_are_reported_not_silently_dropped():
    """静默回落会让人拿到看着正常的数字、却不知道自己的配置没生效 —— 问题要能读出来。"""
    table = PriceTable({"m1": {"input": 1.0}, "m2": {"input": 1.0, "output": 2.0}})

    assert table.price_of("m1") is None                   # 缺 output —— 只丢这一条
    assert table.price_of("m2") == Price(input=1.0, output=2.0, origin="override")
    assert len(table.warnings) == 1 and "m1" in table.warnings[0]


def test_an_empty_override_changes_nothing():
    assert PriceTable("").price_of("fake") is not None
    assert PriceTable(None).price_of("fake") is not None
    assert PriceTable("").warnings == []


# ---------- 接进问答链路 ----------

class UsageLLM:
    """非假模型，流式跑完给出 provider usage。"""

    is_fake = False

    def __init__(self, model: str):
        self.model = model
        self.last_usage = None

    def stream(self, messages):
        yield "答案"
        self.last_usage = {"prompt_tokens": 2000, "completion_tokens": 1000}


def _setup(client, name: str, *, model: str, table: PriceTable | None = None):
    import app.api.deps as deps
    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = build_runtime()
    rt.llm = UsageLLM(model)
    if table is not None:
        rt.price_table = table
    deps._runtime = rt
    return uid, kb, user, db, rt


def test_the_stored_record_carries_the_cost_and_its_price_note(client):
    from app.services import chat_service

    uid, kb, user, db, rt = _setup(client, "price_row", model="m1", table=TABLE)
    try:
        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert rows[0]["cost"] == 0.004
    assert "m1" in rows[0]["price_note"]
    assert out["trace"]["usage"]["cost"] == 0.004        # 跟着 trace / done 事件一起出去


def test_a_record_with_an_unknown_model_stores_no_cost(client):
    from app.services import chat_service

    uid, kb, user, db, rt = _setup(client, "price_unknown", model="没配过的模型")
    try:
        chat_service.answer(db, rt, user, kb, "文档里写了什么？")
        rows = rt.usage_store.list(uid)
    finally:
        db.close()

    assert rows[0]["cost"] is None                       # 存 NULL，不是 0
    assert "单价未知" in rows[0]["price_note"]


# ---------- 老库升级：新增列要补上 ----------

def test_an_existing_usage_table_gains_the_new_columns(tmp_path):
    """老库的 usage_records 没有 cost / price_note —— 补列漏了就会在记账那一刻撞 no such column。"""
    from sqlalchemy import create_engine, inspect, text

    from app.db.migrate import ensure_sqlite_columns

    engine = create_engine("sqlite:///%s" % (tmp_path / "old.db").as_posix())
    with engine.begin() as conn:          # 造一张票 27 时期的旧表
        conn.execute(text("CREATE TABLE usage_records ("
                          "id VARCHAR(32) PRIMARY KEY, user_id VARCHAR(32), model VARCHAR(128), "
                          "input_tokens INTEGER, output_tokens INTEGER, source VARCHAR(16), "
                          "source_note TEXT)"))

    added = ensure_sqlite_columns(engine)

    have = {c["name"] for c in inspect(engine).get_columns("usage_records")}
    assert {"cost", "price_note"} <= have
    assert {"usage_records.cost", "usage_records.price_note"} <= set(added)
