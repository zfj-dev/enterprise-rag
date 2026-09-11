"""成本可见性（票 30 / #37）：管理员看全局、普通用户只看自己；折不出的账不当 0。

聚合是纯函数（app/core/cost.py），取数与权限在 `/metrics/cost` —— 分开测。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.cost import summarize
from tests.helpers import register_and_kb, sse_events

DAY = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _rec(cost, *, uid="u1", at=DAY, tokens=(10, 5), source="provider"):
    return {"user_id": uid, "cost": cost, "created_at": at, "source": source,
            "input_tokens": tokens[0] if tokens else None,
            "output_tokens": tokens[1] if tokens else None}


# ---------- 聚合（纯函数） ----------

def test_totals_by_user_and_by_day():
    out = summarize([
        _rec(0.1, uid="a", at=DAY),
        _rec(0.2, uid="a", at=DAY - timedelta(days=1)),
        _rec(0.4, uid="b", at=DAY),
    ], usernames={"a": "甲", "b": "乙"})

    assert out["records"] == 3
    assert out["total_cost"] == 0.7
    assert [u["key"] for u in out["by_user"]] == ["乙", "甲"]        # 按花费倒序
    assert {d["date"] for d in out["by_day"]} == {"2026-09-10", "2026-09-11"}


def test_the_caliber_travels_with_the_number_into_the_summary():
    """数字聚上去了，**口径不能丢** —— 否则面板上分不清账单、估算还是模拟。"""
    out = summarize([_rec(0.1), {**_rec(0.2), "source": "simulated"},
                     {**_rec(None), "source": "unavailable"}])

    assert out["by_source"] == {"provider": 1, "simulated": 1, "unavailable": 1}


def test_unpriced_records_are_counted_not_treated_as_zero():
    """折不出的账**计条数、不当 0** —— 合计因此是下界，得一并写出来。"""
    out = summarize([_rec(0.1), _rec(None), _rec(None)])

    assert out["total_cost"] == 0.1
    assert out["unpriced"] == 2


def test_when_nothing_can_be_priced_the_total_is_none_not_zero():
    """一笔都折不出 → **None**，不是 0。0 元看起来像「这段时间没花钱」。"""
    out = summarize([_rec(None), _rec(None)])

    assert out["total_cost"] is None
    assert out["unpriced"] == 2
    assert out["by_user"][0]["cost"] is None


def test_no_records_reports_nothing_rather_than_zeros():
    out = summarize([])

    assert out["records"] == 0
    assert out["total_cost"] is None and out["input_tokens"] is None
    assert out["by_user"] == [] and out["by_day"] == []


def test_naive_timestamps_from_sqlite_are_bucketed_without_blowing_up():
    """sqlite 读回来的时间戳不带时区 —— 直接拿去格式化会带上 UTC 偏移、分桶串天。"""
    out = summarize([_rec(0.1, at=DAY.replace(tzinfo=None))])

    assert out["by_day"][0]["date"] == "2026-09-11"


def test_records_without_tokens_are_counted_as_unavailable():
    out = summarize([_rec(0.1, tokens=(10, 5)), _rec(0.1, tokens=None)])

    assert out["input_tokens"] == 10
    assert out["token_unavailable"] == 1          # 合计同样是下界


# ---------- 接口：谁能看到谁 ----------

def _seeded(client, name: str):
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    return H, uid, kb, user, db, get_runtime()


def _spend(rt, uid, cost, *, model="m1"):
    rt.usage_store.add(uid, {"model": model, "input_tokens": 10, "output_tokens": 5,
                             "source": "provider", "source_note": "", "cost": cost,
                             "price_note": ""})


def test_a_plain_user_sees_only_themselves(client, monkeypatch):
    """普通用户只看自己 —— 连「按人」那一列都不给（免得从别人的名字推用量）。"""
    H, uid, kb, user, db, rt = _seeded(client, "cost_self")
    other_H, other_uid, _ = register_and_kb(client, "cost_other")
    try:
        _spend(rt, uid, 0.5)
        _spend(rt, other_uid, 99.0)               # 别人的钱，一滴都不该露出来

        out = client.get("/api/v1/metrics/cost", headers=H).json()
    finally:
        db.close()

    assert out["group"] == "self"
    assert out["total_cost"] == 0.5
    assert out["records"] == 1
    assert out["by_user"] == []
    assert "99" not in str(out)


def test_an_admin_sees_the_global_breakdown(client):
    """管理员看全局：总额、按人（含用户名）、按天。"""
    H, uid, kb, user, db, rt = _seeded(client, "cost_admin")
    other_H, other_uid, _ = register_and_kb(client, "cost_admin2")
    try:
        db.query(type(user)).filter(type(user).id == uid).update({"role": "admin"})
        db.commit()
        _spend(rt, uid, 0.5)
        _spend(rt, other_uid, 0.25)

        out = client.get("/api/v1/metrics/cost", headers=H).json()
    finally:
        db.close()

    assert out["group"] == "global"
    assert out["total_cost"] == 0.75
    assert len(out["by_user"]) == 2
    assert any(u["key"] == "cost_admin2" for u in out["by_user"])   # 按人那一列带用户名
    assert out["by_day"] and out["by_day"][0]["date"]


def test_unpriced_records_make_the_note_say_lower_bound(client):
    H, uid, kb, user, db, rt = _seeded(client, "cost_unpriced")
    try:
        _spend(rt, uid, 0.5)
        _spend(rt, uid, None)                     # 单价未知那一笔

        out = client.get("/api/v1/metrics/cost", headers=H).json()
    finally:
        db.close()

    assert out["total_cost"] == 0.5
    assert out["unpriced"] == 1
    assert "下界" in out["note"]


def test_with_no_usage_the_summary_says_nothing_rather_than_zero(client):
    H, uid, kb, user, db, rt = _seeded(client, "cost_empty")
    db.close()

    out = client.get("/api/v1/metrics/cost", headers=H).json()

    assert out["records"] == 0
    assert out["total_cost"] is None


def test_the_demo_mode_produces_numbers_instead_of_unavailable(client):
    """演示模式也演示得起来（票面第 5 条）：假模型**自报**模拟用量，口径里写明不是账单。"""
    H, uid, kb, user, db, rt = _seeded(client, "cost_demo")
    db.close()
    client.post("/api/v1/chat/stream", headers=H,
                json={"kb_id": kb, "question": "文档里写了什么？", "stream": True})

    out = client.get("/api/v1/metrics/cost", headers=H).json()

    assert out["records"] == 1
    assert out["input_tokens"] and out["input_tokens"] > 0
    assert out["output_tokens"] and out["output_tokens"] > 0
    assert out["token_unavailable"] == 0
    assert out["total_cost"] == 0.0            # 假模型单价就是 0（构造上的事实）
    assert out["by_source"] == {"simulated": 1}         # 口径跟着数字进了摘要
    assert "模拟口径" in out["note"] and "不是账单" in out["note"]   # 面板上就看得出来
    assert "自然窗口" in out["window"] and out["since"]
