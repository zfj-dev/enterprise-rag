"""额度判定与硬拦（票 29）：按用户、按自然窗口累计**费用**，到达额度上限就拒绝新请求。

**为什么叫「额度」而不是「预算」**：[CONTEXT.md](../../CONTEXT.md) 里的**上下文预算**指的是
读取侧能带多少 token（与输出上限相对），是另一回事。同一个词不能既指 token 又指钱 ——
所以这里一律用**额度**（费用口径），与 `context_token_budget` 区分开。

判定是**纯函数**（窗口起点 + 已累计费用 vs 上限），无副作用、可确定性测试 ——
「恰好等于上限」「跨窗口」「没设上限」「开关关闭」这些边角都钉得死。

三条边角写死在 spec 0005 里：
- 判定发生在**生成开始前**，依据的是**此前已累计**的用量 —— 本次请求的用量完成后才入账，
  所以拦截只对**下一个**请求生效，**已在进行中的流不会被打断**；
- **管理员豁免**（免得运维把自己锁在系统外面），但其用量照常记录与展示；
- 窗口按**自然**日 / 月滚动（不是「最近 24 小时」）—— 用户问「这个月还剩多少」时想的是自然月。

费用折不出来的用量（单价未知 / token 量不到）**不折算、也不当 0** —— 累计值因此只是**下界**：
这种账拦不住人（拦不准），但**也不会假装拦住了** —— 每遇上一笔折不出来的，都会 warning 一次，
免得「额度明明开着却永远拦不住」变成一件没人知道的事。
"""
from __future__ import annotations

import logging

from datetime import datetime, timezone
from typing import Sequence

from app.utils.times import as_aware

logger = logging.getLogger(__name__)

WINDOWS = ("day", "month")


def window_start(now: datetime, window: str) -> datetime:
    """自然窗口的起点：日 = 当天 00:00，月 = 当月 1 日 00:00，按**服务器本地时区**。"""
    if window not in WINDOWS:
        raise ValueError("窗口只支持 %s，收到：%r" % (" / ".join(WINDOWS), window))
    local = now.astimezone()
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.replace(day=1) if window == "month" else start


def spent_in_window(records: Sequence[dict], since: datetime) -> tuple[float, int]:
    """窗口内已累计的费用（元）+ **折不出费用的条数**。

    `cost` 为 None 的条目不进合计、也不当 0 —— 合计因此是**下界**，第二个数让调用方
    知道这个下界漏了几笔。
    """
    total = 0.0
    unknown = 0
    for r in records:
        created = r.get("created_at")
        if created is not None and as_aware(created) < as_aware(since):
            continue
        cost = r.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total += float(cost)
        else:
            unknown += 1
    return total, unknown


def over_quota(spent: float, limit: float | None) -> bool:
    """到没到上限。**恰好等于也算到** —— 票据写的是「到达上限后拒绝」，不是「超过才拒」。"""
    if limit is None or limit <= 0:
        return False            # 没设上限 = 不拦
    return spent >= limit


def quota_message(spent: float, limit: float, window: str, unknown: int) -> str:
    """拒绝时的原因文案：哪个窗口、花了多少、上限多少、累计值是什么口径。"""
    label = "今日" if window == "day" else "本月"
    msg = ("%s额度已用完：已累计 %.4f 元，上限 %.4f 元。%s窗口滚动后自动恢复。"
           % (label, spent, limit, label))
    if unknown:
        msg += "（另有 %d 笔费用无法折算、未计入 —— 该累计值是下界）" % unknown
    return msg


def check_quota(store, user, *, enabled: bool, window: str, limit: float | None,
                 now: datetime | None = None) -> str | None:
    """要不要拦这个用户的下一次生成：拦就返回**原因文案**，不拦返回 None。

    纯判定在 `over_quota`；这里只负责取数（按用户下推）与豁免规则。
    """
    if not enabled or getattr(user, "role", "") == "admin":
        return None                       # 关掉开关 / 管理员豁免 —— 用量照记，只是不拦
    now = now or datetime.now().astimezone()
    spent, unknown = spent_in_window(store.list(user.id), window_start(now, window))
    if unknown:
        # 折不出来的账**不代表没花钱**，只代表我们量不到 —— 不说出来，就会变成
        # 「额度明明开着却怎么也拦不住」而没人知道为什么
        logger.warning("额度判定：本窗口有 %d 笔费用折不出来（单价未知 / token 量不到），"
                       "已累计 %.4f 元只是**下界**；用户 %s", unknown, spent, user.id)
    if not over_quota(spent, limit):
        return None
    return quota_message(spent, limit, window, unknown)
