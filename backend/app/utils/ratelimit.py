"""进程内滑动窗口限流。

四处共用同一套形状，只有窗口/上限/键不同：
- `api/v1/auth.py`：按**用户名**限登录、按**客户端 IP** 限注册；
- `api/v1/chat.py`：按**用户**限问答频率；
- `main.py`：按**客户端 IP** 限前端错误上报（那个端点必须未鉴权，只能用 IP）。

**已知边界**（与部署形态绑定，别当成 bug）：计数只在**本进程内** —— 多 worker 时各算各的，
要跨进程得换 Redis（配置里已有 `redis_url`）。调用方各自也在注释里写了这条。
"""
from __future__ import annotations

import threading
import time

# 实例登记表，**只给测试用**：`clear_all()` 一把清空，省得每新增一个限流器就得记得去
# `tests/conftest.py` 补一行清理（漏了就是计数跨用例累积、测试莫名 429）。
_ALL: list["SlidingWindowLimiter"] = []


class SlidingWindowLimiter:
    """窗口内允许 `limit` 次的滑动窗口计数器。

    `limit` 由**每次调用**传入而不是构造时固定：上限来自配置，而配置在测试里会被改。
    """

    def __init__(self, window_seconds: float, max_keys: int = 4096) -> None:
        self.window = window_seconds
        # 键是攻击者可控的（用户名 / IP）：不清就是个只涨不跌的内存泄漏
        # （拿一万个随机用户名登录一万次就够了）。
        self.max_keys = max_keys
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        _ALL.append(self)

    @property
    def table(self) -> dict[str, list[float]]:
        """底层计数表。给测试与既有调用方清理用（`conftest` 会 `.clear()` 它）。"""
        return self._hits

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()

    def allow(self, key: str, limit: int) -> bool:
        """窗口内允许 `limit` 次；超限返回 False（并把该键的时间戳收敛）。"""
        now = time.time()
        with self._lock:
            ts = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(ts) >= limit:
                self._hits[key] = ts
                return False
            ts.append(now)
            if len(self._hits) >= self.max_keys and key not in self._hits:
                self._prune(now)
            self._hits[key] = ts
            return True

    def _prune(self, now: float) -> None:
        """先清过期的窗口；还满就按「最后尝试时间」丢掉最旧的一半。

        只有过期清理是不够的：键全部活跃时（万箭齐发的随机用户名）一个也清不掉，
        表会突破 `max_keys` 无上限地涨。调用方须已持锁。
        """
        for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
            self._hits.pop(k, None)
        if len(self._hits) >= self.max_keys:
            stale = sorted(self._hits, key=lambda k: self._hits[k][-1])
            for k in stale[: self.max_keys // 2]:
                self._hits.pop(k, None)


def clear_all() -> None:
    """清空**所有**限流器的计数（测试用）。"""
    for limiter in _ALL:
        limiter.clear()
