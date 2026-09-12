"""额度判定 + 硬拦（票 29 / #36）：按用户、按自然窗口累计**费用**，到达上限就拒绝**新**请求。

判定是纯函数（窗口起点 + 已累计 vs 上限）—— 边角都钉得死；豁免与开关在调用方。
（叫「额度」不叫「预算」：CONTEXT.md 里「上下文预算」指 token，是另一回事。）
"""
from __future__ import annotations

import logging

from datetime import datetime, timedelta, timezone

import pytest

from app.core.quota import (check_quota, over_quota, quota_message, spent_in_window, window_start)
from tests.helpers import register_and_kb, sse_events

DAY = datetime(2026, 9, 11, 15, 30, tzinfo=timezone.utc)


class FakeUser:
    def __init__(self, uid="u1", role="viewer"):
        self.id = uid
        self.role = role


class FakeStore:
    def __init__(self, records):
        self._records = records

    def list(self, user_id):
        return list(self._records)


def _rec(cost, *, at=None, **extra):
    return {"cost": cost, "created_at": at or DAY, **extra}


# ---------- 窗口 ----------

def test_the_window_starts_at_the_natural_day_and_month():
    day = window_start(DAY, "day")
    month = window_start(DAY, "month")

    assert (day.hour, day.minute, day.second) == (0, 0, 0)
    assert day.day == 11
    assert (month.day, month.hour) == (1, 0)


def test_an_unknown_window_name_is_rejected_loudly():
    """窗口名写错就当场报错 —— 别静默按某个默认值滚，那会让上限口径变成猜的。"""
    with pytest.raises(ValueError):
        window_start(DAY, "week")


# ---------- 累计 ----------

def test_only_records_inside_the_window_are_counted():
    since = window_start(DAY, "day")
    records = [_rec(0.1), _rec(0.2), _rec(0.9, at=DAY - timedelta(days=1))]

    total, unknown = spent_in_window(records, since)

    assert round(total, 6) == 0.3 and unknown == 0


def test_unknown_costs_are_counted_as_unknown_not_as_zero():
    """折不出的费用**不进合计、也不当 0** —— 合计因此是下界，漏了几笔要说出来。"""
    total, unknown = spent_in_window([_rec(0.1), _rec(None), _rec(None)], window_start(DAY, "day"))

    assert round(total, 6) == 0.1
    assert unknown == 2


def test_naive_timestamps_from_sqlite_are_compared_without_blowing_up():
    """sqlite 读回来的 `created_at` **没有时区**，而窗口起点有 —— 直接比会 TypeError。

    真机上就是这条路径：只要有过一笔用量，下一次问答就会 500。
    """
    since = window_start(DAY, "day")
    naive_in = DAY.replace(tzinfo=None)                          # 窗口内（naive）
    naive_out = (DAY - timedelta(days=2)).replace(tzinfo=None)   # 窗口外（naive）

    total, unknown = spent_in_window([_rec(0.1, at=naive_in), _rec(0.9, at=naive_out)], since)

    assert round(total, 6) == 0.1 and unknown == 0


# ---------- 判定（边角写死） ----------

def test_exactly_at_the_limit_counts_as_over():
    """票据写的是「**到达**上限后拒绝」—— 恰好等于就得拦，不是「超过才拦」。"""
    assert over_quota(1.0, 1.0) is True
    assert over_quota(0.999, 1.0) is False
    assert over_quota(2.0, 1.0) is True


def test_no_limit_configured_never_blocks():
    assert over_quota(999.0, None) is False
    assert over_quota(999.0, 0.0) is False
    assert over_quota(999.0, -1.0) is False


def test_the_reason_says_the_window_the_spend_and_the_limit():
    msg = quota_message(1.5, 1.0, "day", 0)

    assert "今日" in msg and "1.5000" in msg and "1.0000" in msg
    assert "自动恢复" in msg                 # 说清怎么办，别只丢一个「不行」


def test_the_reason_admits_when_the_total_is_only_a_lower_bound():
    msg = quota_message(1.5, 1.0, "day", 2)

    assert "2 笔" in msg and "下界" in msg


# ---------- 豁免与开关 ----------

def test_an_admin_is_exempt():
    store = FakeStore([_rec(100.0)])

    assert check_quota(store, FakeUser(role="admin"), enabled=True,
                        window="day", limit=1.0, now=DAY) is None


def test_the_switch_off_never_blocks():
    store = FakeStore([_rec(100.0)])

    assert check_quota(store, FakeUser(), enabled=False,
                        window="day", limit=1.0, now=DAY) is None


def test_over_quota_returns_the_reason():
    store = FakeStore([_rec(2.0)])

    msg = check_quota(store, FakeUser(), enabled=True, window="day", limit=1.0, now=DAY)

    assert msg and "今日额度已用完" in msg


# ---------- 接进问答端点 ----------

def _seeded(client, name: str):
    from app.api.deps import get_runtime
    from app.config import get_settings
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    rt = get_runtime()
    return H, uid, kb, user, db, rt, get_settings()


def _spend(rt, uid, cost):
    rt.usage_store.add(uid, {"model": "m1", "input_tokens": 1, "output_tokens": 1,
                             "source": "provider", "source_note": "", "cost": cost,
                             "price_note": ""})


def test_a_request_over_quota_is_rejected_with_a_clear_reason(client, monkeypatch):
    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_over")
    try:
        monkeypatch.setattr(conf, "quota_enabled", True)
        monkeypatch.setattr(conf, "quota_limit", 1.0)
        _spend(rt, uid, 1.0)                      # 恰好到上限

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 402                   # 额度用尽，不是 429 / 403
    assert "额度已用完" in r.json()["detail"]


def test_a_request_under_the_limit_goes_through_untouched(client, monkeypatch):
    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_under")
    try:
        monkeypatch.setattr(conf, "quota_enabled", True)
        monkeypatch.setattr(conf, "quota_limit", 10.0)
        _spend(rt, uid, 0.5)

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
        kinds = [e["type"] for e in sse_events(r.text)]
    finally:
        db.close()

    assert r.status_code == 200
    assert kinds[0] == "sources" and kinds[-1] == "done"    # 没被中途打断


def test_the_block_only_hits_the_next_request(client, monkeypatch):
    """已在跑的那次照常走完；额度用完之后，**下一个**请求才被拦。"""
    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_next")
    try:
        monkeypatch.setattr(conf, "quota_enabled", True)
        monkeypatch.setattr(conf, "quota_limit", 1.0)
        _spend(rt, uid, 0.4)

        first = client.post("/api/v1/chat/stream", headers=H,
                            json={"kb_id": kb, "question": "问", "stream": True})
        assert first.status_code == 200                       # 这次没被拦，也没被打断

        _spend(rt, uid, 1.0)                                  # 这次问答的用量入账后超了
        second = client.post("/api/v1/chat/stream", headers=H,
                             json={"kb_id": kb, "question": "再问", "stream": True})
    finally:
        db.close()

    assert second.status_code == 402


def test_the_admin_is_exempt_but_still_recorded(client, monkeypatch):
    """管理员不受拦（免得把自己锁在外面），但用量**照常记录**。"""
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_admin")
    try:
        db.query(User).filter(User.id == uid).update({"role": "admin"})
        db.commit()
        user = db.query(User).filter(User.id == uid).first()   # 取回提升后的角色
        monkeypatch.setattr(conf, "quota_enabled", True)
        monkeypatch.setattr(conf, "quota_limit", 1.0)
        _spend(rt, uid, 99.0)

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 200
    assert len(rt.usage_store.list(uid)) == 2                 # 预置那笔 + 这次问答记的那笔


def test_the_switch_off_keeps_behaviour_as_today(client, monkeypatch):
    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_off")
    try:
        monkeypatch.setattr(conf, "quota_enabled", False)
        monkeypatch.setattr(conf, "quota_limit", 0.01)
        _spend(rt, uid, 99.0)

        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 200


def test_an_unpriced_record_is_surfaced_even_when_still_under_the_limit(caplog):
    """折不出来的账**不代表没花钱** —— 不说出来，就成了「额度开着却怎么也拦不住」而没人知道。"""
    store = FakeStore([_rec(None), _rec(None)])

    with caplog.at_level(logging.WARNING):
        assert check_quota(store, FakeUser(), enabled=True, window="day",
                           limit=1.0, now=DAY) is None        # 没到上限：不拦

    assert "下界" in caplog.text                              # 但必须说出来


def test_the_guard_runs_once_at_entry_not_during_the_stream(client, monkeypatch):
    """判定只在**进入请求时**做一次 —— 流一旦开始就不会被中途掐断（构造上的保证，不只是没测到）。"""
    import app.api.v1.chat as chat_api

    calls: list = []
    monkeypatch.setattr(chat_api, "check_quota", lambda *a, **k: calls.append(1))

    H, uid, kb, user, db, rt, conf = _seeded(client, "quota_once")
    try:
        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "问", "stream": True})
    finally:
        db.close()

    assert r.status_code == 200
    assert len(calls) == 1                                    # 整条流里只判了一次
