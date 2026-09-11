"""per-query 用量记账（票 27）：每次生成记一条用量，**口径来源必须写清楚**。

优先级：**provider usage（账单口径）> 本地分词器（估算口径）> 不可用**。
两者皆无时标记「不可用」而**不记 0** —— 0 是「没花 token」，与「量不到」是两回事，
混起来会让「这次花了 0」看起来像结论（与 Spec 0003 的降级诚实性同源）。

存储按用户隔离（与记忆、检索同源下推），供后续的费用折算 / 预算 / 可见性使用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.db.session import SessionLocal
from app.models.entities import UsageRecord

SOURCE_PROVIDER = "provider"        # provider 返回的 usage —— 与账单一致
SOURCE_LOCAL = "local"              # 本地分词器数的 —— 估算
SOURCE_UNAVAILABLE = "unavailable"  # 两者皆无 —— 不记 0


def as_int(value):
    """把 provider 给的数收成 int —— bool 也是 int，别把 True 当成 1 个 token。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def build_usage(*, prompt_text: str, answer_text: str, model: str,
                provider_usage: dict | None = None, token_counter=None) -> dict:
    """造一条用量记录（纯函数，依赖全部注入）。

    - provider usage 两侧齐全 → **账单口径**，直接用它的数
    - 否则有真实分词器 → **估算口径**，本地数（分词器名写进口径说明）
    - 都没有 → **不可用**，token 记 None 并写明原因

    半边的 provider usage **不补齐**：缺的那侧补个 0，等于编了一个数出来。
    """
    usage = provider_usage or {}
    p_in = as_int(usage.get("prompt_tokens"))
    p_out = as_int(usage.get("completion_tokens"))
    if p_in is not None and p_out is not None:
        return {"model": model, "input_tokens": p_in, "output_tokens": p_out,
                "source": SOURCE_PROVIDER,
                "source_note": "账单口径：provider 返回的 usage"}

    label = getattr(token_counter, "label", "") or ""
    if label:
        return {"model": model,
                "input_tokens": token_counter.count(prompt_text or ""),
                "output_tokens": token_counter.count(answer_text or ""),
                "source": SOURCE_LOCAL,
                "source_note": "估算口径：本地分词器 %s" % label}

    why = "provider 未返回 usage" + ("（只回了半边，不补齐）" if usage else "")
    return {"model": model, "input_tokens": None, "output_tokens": None,
            "source": SOURCE_UNAVAILABLE,
            "source_note": "不可用：%s，也没有真实分词器 —— 拿不到就不记 0" % why}


class UsageStore(ABC):
    """按用户存取用量记录。全部按 user 下推过滤，取不到他人的。"""

    @abstractmethod
    def add(self, user_id: str, record: dict, *, session_id: str = "",
            message_id: str = "") -> None: ...

    @abstractmethod
    def list(self, user_id: str) -> list[dict]: ...


class InMemoryUsageStore(UsageStore):
    """内存实现：测试隔离用。"""

    def __init__(self) -> None:
        self._by_user: dict[str, list[dict]] = {}
        self._seq = 0

    def add(self, user_id, record, *, session_id="", message_id=""):
        self._seq += 1
        self._by_user.setdefault(user_id, []).append(
            {"id": "u%d" % self._seq, "user_id": user_id, "session_id": session_id,
             "message_id": message_id, **record})

    def list(self, user_id):
        return list(self._by_user.get(user_id, []))


class DbUsageStore(UsageStore):
    """落库实现。自己开 session —— 记账的调用点未必在请求期的 session 里。"""

    def add(self, user_id, record, *, session_id="", message_id=""):
        db = SessionLocal()
        try:
            db.add(UsageRecord(user_id=user_id, model=record.get("model") or "",
                               input_tokens=record.get("input_tokens"),
                               output_tokens=record.get("output_tokens"),
                               source=record.get("source") or "",
                               source_note=record.get("source_note") or "",
                               cost=record.get("cost"),
                               price_note=record.get("price_note") or "",
                               source_session_id=session_id or "",
                               source_message_id=message_id or ""))
            db.commit()
        finally:
            db.close()

    def list(self, user_id):
        db = SessionLocal()
        try:
            # created_at 由 ORM 侧单调默认值保证严格递增；id 只是同秒旧行的确定性兜底
            rows = (db.query(UsageRecord).filter(UsageRecord.user_id == user_id)
                    .order_by(UsageRecord.created_at.asc(), UsageRecord.id.asc()).all())
            return [{"id": r.id, "user_id": r.user_id, "model": r.model,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "source": r.source, "source_note": r.source_note,
                     "cost": r.cost, "price_note": r.price_note,
                     "session_id": r.source_session_id, "message_id": r.source_message_id}
                    for r in rows]
        finally:
            db.close()
