"""时间小工具：跨模块共用，别各写一份。"""
from __future__ import annotations

from datetime import datetime, timezone


def as_aware(value: datetime) -> datetime:
    """把时间戳补齐时区再比较 / 分桶。

    sqlite 存 `DateTime(timezone=True)` 时**会把时区丢掉**，读回来是 naive —— 而我们的窗口起点
    是 aware，两者直接比会抛 `TypeError`。写入端本来就是 UTC，所以 naive 一律按 UTC 解释。

    额度判定、成本按天分桶、成本时间窗三处都要这一步，收在这里一份。
    """
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
