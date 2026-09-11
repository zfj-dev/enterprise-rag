"""成本可见性（票 30）：把用量记录聚成一份可读的成本摘要。

**纯函数**（给定记录集合 → 总额 / 按人 / 按天）—— 取数与权限在调用方（API 层），
所以「谁能看到谁」和「数字怎么算」分开测。

折不出的费用**计条数、不当 0**：有折不出的条目时，总额只是**下界**（一并报出来）；
一笔都折不出时总额是 **None**（不是 0 —— 0 元看起来像「没花钱」）。
"""
from __future__ import annotations

from typing import Sequence

from app.utils.times import as_aware


def _record_cost(record: dict) -> float | None:
    """折得出来的费用；**折不出来的返回 None**（bool 也是 int，挡掉）。"""
    cost = record.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        return float(cost)
    return None


def _day_of(record: dict) -> str:
    """记录落在哪一天（本地时区）—— 按天分桶用。"""
    created = record.get("created_at")
    if created is None:
        return ""
    return as_aware(created).astimezone().strftime("%Y-%m-%d")


def _tokens_of(record: dict) -> tuple[int, int] | None:
    i, o = record.get("input_tokens"), record.get("output_tokens")
    if isinstance(i, int) and not isinstance(i, bool) \
            and isinstance(o, int) and not isinstance(o, bool):
        return i, o
    return None


def summarize(records: Sequence[dict], *, usernames: dict | None = None) -> dict:
    """聚合成成本摘要。`usernames` 是 `{user_id: 用户名}` 查表（缺了就只给 id）。"""
    if not records:
        # 一条都没有：什么都报 **None**，别拿 0 冒充「算过了，就是零」
        return {"records": 0, "total_cost": None, "unpriced": 0, "input_tokens": None,
                "output_tokens": None, "token_unavailable": 0, "by_source": {},
                "by_user": [], "by_day": []}

    names = usernames or {}
    total = 0.0
    priced = 0
    tok_in = tok_out = tok_missing = 0
    by_source: dict[str, int] = {}
    by_user: dict[str, dict] = {}
    by_day: dict[str, dict] = {}

    def _bump(holder: dict, key: str, cost: float | None) -> None:
        b = holder.setdefault(key, {"cost": 0.0, "priced": 0, "unpriced": 0, "records": 0})
        b["cost"] += cost or 0.0
        b["priced" if cost is not None else "unpriced"] += 1
        b["records"] += 1

    for r in records:
        cost = _record_cost(r)
        tokens = _tokens_of(r)
        if cost is None:
            pass
        else:
            total += cost
            priced += 1
        if tokens is None:
            tok_missing += 1
        else:
            tok_in += tokens[0]
            tok_out += tokens[1]
        src = r.get("source") or "unknown"
        by_source[src] = by_source.get(src, 0) + 1
        _bump(by_user, r.get("user_id") or "", cost)
        _bump(by_day, _day_of(r), cost)

    def _rows(holder: dict, key_field: str, label_of) -> list[dict]:
        return [{key_field: key, "key": label_of(key),
                 "cost": round(b["cost"], 8) if b["priced"] else None,
                 "priced": b["priced"], "unpriced": b["unpriced"], "records": b["records"]}
                for key, b in holder.items()]

    users = _rows(by_user, "user_id", lambda k: names.get(k) or k)
    users.sort(key=lambda x: (x["cost"] is None, -(x["cost"] or 0.0)))
    days = sorted(_rows(by_day, "date", lambda k: k), key=lambda x: x["date"])

    return {
        "records": len(records),
        "total_cost": round(total, 8) if priced else None,   # 一笔都折不出就是 None，不是 0
        "unpriced": sum(1 for r in records if _record_cost(r) is None),
        "input_tokens": None if tok_missing == len(records) else tok_in,
        "output_tokens": None if tok_missing == len(records) else tok_out,
        "token_unavailable": tok_missing,
        "by_user": users,
        "by_day": days,
        "by_source": by_source,   # 账单 / 估算 / 模拟 / 不可用各几笔 —— 口径跟着数字走
    }
