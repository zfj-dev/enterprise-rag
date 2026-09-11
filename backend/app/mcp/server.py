"""把同一组工具挂成 MCP server（stdio）—— 任何 MCP 客户端都能挂载复用。

只暴露 **tools**（不做 resources / prompts）；默认走本地 stdio，不占端口、不出网络。
对外暴露（http/sse）与鉴权见票 16。

这里挂的是 `default_registry()`（不依赖请求上下文的工具，如 Calculator）。
需要请求上下文（kb_id / owner_id）的工具由 `app/mcp/registry.build_registry` 造 ——
把它们一起对外暴露是票 16 的事。
"""
from __future__ import annotations

import asyncio
import json

from app.core.tools import ToolError, ToolRegistry, default_registry

SERVER_NAME = "enterprise-rag"
SERVER_VERSION = "1.0"


def list_tools_result(registry: ToolRegistry):
    """「发现工具」的应答。摊成模块级函数 —— 测它不必起真实 MCP 进程。"""
    import mcp.types as types

    return types.ListToolsResult(tools=[
        types.Tool(name=s["name"], description=s["description"],
                   input_schema=s["inputSchema"])
        for s in registry.specs()])


def call_tool_result(registry: ToolRegistry, name: str, arguments: dict | None = None):
    """「调用工具」的应答。入参不合法是**这次调用失败**，不是服务挂了 —— 如实回给客户端。"""
    import mcp.types as types

    try:
        out = registry.call(name, arguments or {})
    except ToolError as e:
        return types.CallToolResult(
            content=[types.TextContent(type="text",
                                       text=json.dumps({"error": str(e)}, ensure_ascii=False))],
            is_error=True)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(out, ensure_ascii=False))])


def build_server(registry: ToolRegistry):
    """按注册表造一个 MCP server。SDK 到这里才 import —— 测试走进程内传输，不必装它。"""
    from mcp.server import Server

    async def on_list_tools(context, params):
        return list_tools_result(registry)

    async def on_call_tool(context, params):
        return call_tool_result(registry, params.name, params.arguments)

    return Server(SERVER_NAME, version=SERVER_VERSION,
                  on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve(registry: ToolRegistry | None = None) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(registry or default_registry())
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
