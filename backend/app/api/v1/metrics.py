"""评估指标 + 成本可见性（票 30）。

成本**复用这一个入口**，不另起观测通道：`GET /metrics/cost` —— 管理员看全局（总额 / 按人 /
按天），普通用户只看自己。数字怎么算在 `app/core/cost.py`（纯函数），这里只管取数与权限。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_runtime, require_admin
from app.config import get_settings
from app.core.container import Runtime
from app.core.cost import summarize, summarize_note
from app.core.quota import window_start
from app.core.schemas import CostSummaryOut, MetricsOut
from app.utils.times import as_aware
from app.models.entities import ChatMessage, User

router = APIRouter(prefix="/metrics", tags=["metrics"])

_MAX_DAYS = 365


@router.get("", response_model=MetricsOut)
def metrics(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    answered = db.query(ChatMessage).filter(ChatMessage.role == "assistant").count()
    return MetricsOut(total_answered=answered)


@router.get("/cost", response_model=CostSummaryOut)
def cost_summary(days: int | None = None, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db), rt: Runtime = Depends(get_runtime)):
    """成本摘要。**管理员看全局，普通用户只看自己** —— 权限在这一处收口，不指望前端不问。

    默认按**自然窗口**（与额度同一个口径：日 / 月，可配）—— 「这个月烧了多少」问的就是自然月；
    想拉一段历史就传 `days=N`（滚动 N 天）。
    """
    now = datetime.now().astimezone()
    if days is None:
        since = window_start(now, get_settings().quota_window)
        window_label, effective_days = "自然窗口（%s）" % get_settings().quota_window, None
    else:
        effective_days = max(1, min(days, _MAX_DAYS))
        since = now - timedelta(days=effective_days)
        window_label = "最近 %d 天" % effective_days
    admin = user.role == "admin"

    # 范围由服务端注入：普通用户拿到的就是自己那批，客户端传什么都没用
    records = rt.usage_store.list_all() if admin else rt.usage_store.list(user.id)
    records = [r for r in records if _within(r, since)]

    names = {u.id: u.username for u in db.query(User).all()} if admin else {}
    payload = summarize(records, usernames=names)
    if not admin:
        payload["by_user"] = []          # 「按人」那一列只属于管理员
        payload["group"] = "self"
    else:
        payload["group"] = "global"
    payload["days"] = effective_days
    payload["window"] = window_label
    payload["since"] = since.strftime("%Y-%m-%d %H:%M")
    payload["note"] = _note(payload)
    return CostSummaryOut(**payload)


def _within(record: dict, since: datetime) -> bool:
    """记录是否落在时间窗内。sqlite 读回来的时间戳不带时区（写入端本来就是 UTC）。"""
    created = record.get("created_at")
    return True if created is None else as_aware(created) >= since


def _note(payload: dict) -> str:
    """口径写清：数字到了面板上，**口径不能丢** —— 账单 / 估算 / 模拟必须跟着一起显示；
    并且**标明这是我方统计、不是厂商余额**（票 35 要求两个数字各自的来源分别写明）。"""
    parts = ["我方统计用量（本地记账按单价折算，不是厂商余额）", summarize_note(payload)]
    return "；".join(p for p in parts if p)
