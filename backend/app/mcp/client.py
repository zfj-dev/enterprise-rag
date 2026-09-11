"""代理看工具的那条缝（spec 0002：代理核心**不直接建 MCP 连接**，只认这个接口）。

- `InProcessTransport` —— 直调同一个注册表。测试与内部使用，**不起任何真实 MCP 进程、不联网**。
- `StdioMCPClient`    —— 经官方 mcp SDK 连本地 stdio server（生产 / 对外挂载用）。
  SDK 惰性 import：没装也不影响测试。
"""
from __future__ import annotations

import asyncio
import json
import threading

from abc import ABC, abstractmethod


class ToolTransport(ABC):
    """list_tools / call_tool —— 工具集怎么到达代理，代理不关心。"""

    @abstractmethod
    def list_tools(self) -> list: ...

    @abstractmethod
    def call_tool(self, name: str, arguments: dict | None = None) -> dict: ...


class InProcessTransport(ToolTransport):
    """直调注册表 —— 工具实现只有一份，这里不再抄一遍。"""

    def __init__(self, registry):
        self._registry = registry

    def list_tools(self) -> list:
        return self._registry.specs()

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self._registry.call(name, arguments)


def _result_to_dict(result) -> dict:
    """把 MCP 的 CallToolResult 摊平成 dict：优先用结构化内容，否则解析文本里的 JSON。

    失败时**保留服务端给的错误信息**，只在旁边补一个 `is_error` 标志 ——
    不能拿布尔把原因冲掉，否则调用方只知道「失败了」却不知道为什么。
    """
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and structured:
        out = dict(structured)
    else:
        texts = [c.text for c in (getattr(result, "content", None) or [])
                 if getattr(c, "type", "") == "text"]
        payload = texts[0] if texts else ""
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = {"text": payload}
        out = parsed if isinstance(parsed, dict) else {"result": parsed}
    if getattr(result, "is_error", False):
        out["is_error"] = True
    return out


class StdioMCPClient(ToolTransport):
    """连本地 stdio MCP server 的同步外观客户端。

    会话跑在自己的事件循环线程里（anyio 的取消作用域要求进入/退出在同一任务），
    外面的同步方法通过 run_coroutine_threadsafe 把请求递进去。
    """

    def __init__(self, command: str, args: list, cwd: str | None = None,
                 timeout: float = 30.0):
        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        self._stdio_client = stdio_client
        self._ClientSession = ClientSession
        self._params = StdioServerParameters(command=command, args=list(args), cwd=cwd)
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._started = threading.Event()
        self._error: Exception | None = None
        self._session = None
        self._stop = None
        self._done = threading.Event()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop)

        if not self._started.wait(timeout):
            self.close()
            raise TimeoutError("MCP server 启动超时")
        if self._error is not None:
            error = self._error
            self.close()
            raise error

    async def _serve(self) -> None:
        from contextlib import AsyncExitStack

        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._stdio_client(self._params))
                session = await stack.enter_async_context(self._ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._started.set()
                await self._stop.wait()
        except Exception as e:   # noqa: BLE001 —— 启动失败要能传到构造它的同步调用方
            self._error = e
            self._started.set()
        finally:
            self._session = None
            self._done.set()      # 退栈完成（子进程已关），close() 据此继续收尾

    def _call(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(self._timeout)

    def list_tools(self) -> list:
        got = self._call(self._session.list_tools())
        return [t.model_dump(by_alias=True, exclude_none=True) for t in got.tools]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return _result_to_dict(self._call(self._session.call_tool(name, arguments or {})))

    def close(self) -> None:
        """让会话协程自己退出（取消作用域必须在同一任务里退出，子进程才会被关掉），再收线程。"""
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._done.wait(timeout=self._timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=self._timeout)
        self._loop.close()
