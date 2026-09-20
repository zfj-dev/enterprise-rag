"""请求体大小上限（ASGI 中间件）。

**为什么不能只靠字段级的 `max_length`**：Pydantic 的校验发生在 Starlette 把整个 body
读进内存**之后**。所以「给未鉴权的 `/api/v1/client-error` 发 8MB JSON」照样能把进程
撑爆 —— 那是实测过的（安全审查 F1）。真正省内存的只有两条路子：

1. 反代上封顶（`deploy/Caddyfile` 的 `request_body`）—— 生产走反代时的第一道。
   注意反代那边只设**一个总上限**（取上传那一档），按路径细分交给这一层：Caddy 对同一
   site block 里的多条 `request_body` 是**串联**而非覆盖，全局那条会连放宽的路径一起挡掉。
2. 在这里**读之前**封顶 —— 直连 uvicorn 的部署形态（本机 / 局域网）只有这一道，
   而按路径放宽的细粒度限制也只有这一层做。

两条都要有：反代挡的是外面，这一层挡的是「有人绕过反代直连 8000」以及「没上反代」。
"""
from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE_BODY = json.dumps({"detail": "请求体过大"}, ensure_ascii=False).encode("utf-8")

# 只对这么小的上限做「缓冲后判断」。上传那种大上限**不缓冲**：那会把 Starlette 本来
# 要 spool 到磁盘的几十 MB 搬进内存，等于自己造一个新的 OOM 面 —— 大请求交给
# Content-Length 预判 + 业务层的边读边限长（`documents._read_capped`）。
_BUFFER_MAX = 2 * 1024 * 1024


def _declared_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key == b"content-length":
            try:
                return int(value)
            except ValueError:
                return 0        # 畸形的 Content-Length：当作 0，交给后面的实际计数
    return None


class BodySizeLimitMiddleware:
    """按字节数给请求体封顶，超限在**读进内存之前**返回 413。

    `overrides` 是「路径前缀 -> 单独上限」，用于上传这类天然很大的接口；
    按前缀长度降序匹配，避免 `/api/v1/documents` 抢掉更具体的前缀。
    """

    def __init__(self, app: ASGIApp, max_bytes: int, overrides: dict[str, int] | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self._overrides = tuple(sorted((overrides or {}).items(), key=lambda kv: -len(kv[0])))

    def limit_for(self, path: str) -> int:
        for prefix, n in self._overrides:
            if path.startswith(prefix):
                return n
        return self.max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.limit_for(scope.get("path", ""))

        # 1) 有 Content-Length 就先判：不花一块内存，也是绝大多数客户端的实际情况。
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await self._reject(send)
            return

        # 2) 大上限的路径（上传）**不缓冲**，直接放行：见 `_BUFFER_MAX` 的说明。
        if limit > _BUFFER_MAX:
            await self.app(scope, receive, send)
            return

        # 3) 小上限的路径：边收边数，最多缓冲 limit 字节（≈1MB，可忽略）。
        #    没有 Content-Length 的 chunked 请求只有这条能挡。
        chunks: list[bytes] = []
        total = 0
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                # 客户端还没发完就断了：按原样往下传，别自己造一个 body
                await self.app(scope, _replay_disconnect(), send)
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            total += len(body)
            if total > limit:
                await self._reject(send)
                return
            chunks.append(body)
            more = message.get("more_body", False)

        await self.app(scope, _replay_body(b"".join(chunks), receive), send)

    async def _reject(self, send: Send) -> None:
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                (b"content-length", str(len(_TOO_LARGE_BODY)).encode())]})
        await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})


def _replay_body(body: bytes, original: Receive) -> Receive:
    """把已经收完的 body 重新喂给下游，之后**把 receive 交还给真正的那个**。

    ⚠️ 后续调用**绝不能**直接回 `http.disconnect`：`StreamingResponse`（SSE 就用它）
    会一边发响应、一边 `listen_for_disconnect(receive)`；收到 disconnect 就认为
    「客户端已经走了」，立刻取消整个发送任务 —— 结果是响应一个字节都发不出去，
    外层只看到 `RuntimeError: No response returned.`（这个坑是实测踩出来的）。
    真正的 receive 在 body 收完后会**阻塞**到客户端断开，那才是它期待的行为。
    """
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original()

    return receive


def _replay_disconnect() -> Receive:
    async def receive() -> Message:
        return {"type": "http.disconnect"}

    return receive
