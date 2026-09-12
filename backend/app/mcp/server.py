"""把同一组工具挂成 MCP server —— 任何 MCP 客户端都能挂载复用（票 10 / 票 16）。

只暴露 **tools**（不做 resources / prompts）。传输有两条：

- **stdio（默认）**：本地子进程，不占端口、不出网络 —— 外部客户端直接起这个进程。
- **http（显式开启）**：对外挂载时用；必须同时给出 `MCP_ALLOW_NETWORK=true` 与 `MCP_TOKEN`，
  否则**拒绝启动**；不写 `MCP_HOST` 就只绑 127.0.0.1。

身份与范围由 `--user` / `--kb` 显式配置（见 app/mcp/mount.py）—— 客户端传什么都不看。

挂载示例（Claude Desktop 的 mcpServers；Claude Code 的 .mcp.json 同理）：

    "enterprise-rag": {
      "command": "<backend>/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp.server", "--user", "admin", "--kb", "<kb_id>"],
      "cwd": "<backend>",
      "env": {"DATABASE_URL": "sqlite:///./rag.db"}
    }

环境变量：`MCP_USER` / `MCP_KB_ID`（同 --user / --kb）、`MCP_TRANSPORT`（stdio / http）、
`MCP_HOST` / `MCP_PORT` / `MCP_ALLOW_NETWORK` / `MCP_TOKEN`（对外暴露时才用得上）。

**env 要给全**：MCP SDK 默认只把一份**白名单**环境交给子进程（PATH / TEMP 之类），
数据库地址与模型配置都不在其中 —— 少了它，挂载点会因连不上库当场退出（stdout 是
JSON-RPC 通道，诊断一律走 stderr）。
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys

from app.core.tools import ToolError, ToolRegistry, default_registry

SERVER_NAME = "enterprise-rag"
SERVER_VERSION = "1.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


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


# ---------- 对外暴露：显式配置 + 鉴权（票 16）----------

def _truthy(raw) -> bool:
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_transport(env: dict | None = None) -> dict:
    """传输与鉴权的**显式**决策。

    默认 stdio（本地、无端口、无网络）。想走网络传输（http）必须同时给出
    `MCP_ALLOW_NETWORK=true` 与 `MCP_TOKEN`，少一个就拒绝启动；反过来，想把宿主名
    绑到本机以外，同样要显式允许 —— 免得「改了环境变量就悄悄开了外网」。
    """
    env = os.environ if env is None else env
    transport = (env.get("MCP_TRANSPORT") or "stdio").strip().lower()
    host = (env.get("MCP_HOST") or "").strip()
    allow = _truthy(env.get("MCP_ALLOW_NETWORK"))
    token = (env.get("MCP_TOKEN") or "").strip()

    if transport == "stdio":
        if host and host.lower() not in _LOCAL_HOSTS and not allow:
            raise ValueError("MCP_HOST=%s 把服务开到了本机以外：请显式设 MCP_ALLOW_NETWORK=true "
                             "并给出 MCP_TOKEN" % host)
        return {"transport": "stdio"}

    if transport != "http":
        # 只认自己真实现的：收下 sse 这个别名却发 streamable-http，等于骗配置的人
        raise ValueError("不支持的传输：%s（可用：stdio / http）" % transport)
    if not allow:
        raise ValueError("走网络传输要显式允许：设 MCP_ALLOW_NETWORK=true")
    if not token:
        raise ValueError("对外暴露必须有鉴权令牌：设 MCP_TOKEN=<随机串>")
    try:
        port = int(env.get("MCP_PORT") or DEFAULT_PORT)
    except (TypeError, ValueError):
        raise ValueError("MCP_PORT 必须是整数，收到 %r" % env.get("MCP_PORT"))
    if not 1 <= port <= 65535:
        raise ValueError("MCP_PORT 要在 1..65535 之间，收到 %d" % port)
    return {"transport": "http", "host": host or DEFAULT_HOST, "port": port, "token": token}


def _is_authorized(headers, token: str) -> bool:
    """请求头里有没有正确的 Bearer 令牌（按字节比，恒定时间）。"""
    wanted = ("Bearer " + token).encode("utf-8")
    for name, value in headers:
        if name.lower() == b"authorization":
            return hmac.compare_digest(bytes(value).strip(), wanted)
    return False


class BearerTokenMiddleware:
    """对外暴露的最低限度鉴权：令牌不对就 401，不放行到 MCP 层。

    只拦 http 请求；lifespan 之类的非 http 消息原样放行，否则服务起不来。
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or _is_authorized(scope.get("headers") or [], self.token):
            await self.app(scope, receive, send)
            return
        body = json.dumps({"error": "unauthorized"}, ensure_ascii=False).encode("utf-8")
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def build_http_app(registry: ToolRegistry, *, host: str, token: str):
    """带鉴权的 HTTP 挂载点。

    host 一并交给 SDK：**只有** 127.0.0.1 / localhost / ::1 时它才会自动开
    DNS-rebinding 保护（校验 Host 头）；绑到别的地址那层保护是关的 —— 那时全靠
    Bearer 令牌兜底，所以「对外」这条路必须先有 allow + token。
    """
    app = build_server(registry).streamable_http_app(host=host)
    app.add_middleware(BearerTokenMiddleware, token=token)
    return app


async def serve_http(registry: ToolRegistry, *, host: str, port: int, token: str) -> None:
    import uvicorn

    config = uvicorn.Config(build_http_app(registry, host=host, token=token),
                            host=host, port=port)
    await uvicorn.Server(config).serve()


def main(argv: list[str] | None = None) -> None:
    """入口：python -m app.mcp.server [--user NAME] [--kb ID]。"""
    import argparse

    parser = argparse.ArgumentParser(description="把工具集挂成 MCP server（默认本地 stdio）")
    parser.add_argument("--user", default=os.environ.get("MCP_USER"),
                        help="以哪个用户的身份挂载；不给就只挂不依赖上下文的工具")
    parser.add_argument("--kb", default=os.environ.get("MCP_KB_ID"),
                        help="绑哪个知识库；不给则取该用户的第一个")
    args = parser.parse_args(argv)

    from app.mcp.mount import registry_for_mount

    try:
        config = resolve_transport()
        registry = registry_for_mount(args.user, args.kb)
    except ValueError as e:      # 配置不对就起不来，绝不静默降级
        raise SystemExit("挂载被拒：%s" % e)
    # 注意：stdio 传输的 **stdout 是 JSON-RPC 通道** —— 诊断只能走 stderr，否则握手当场断
    print("[mcp] 传输=%s 工具=%s" % (config["transport"],
                                    ",".join(s["name"] for s in registry.specs())),
          file=sys.stderr, flush=True)
    if config["transport"] == "stdio":
        asyncio.run(serve(registry))
    else:
        asyncio.run(serve_http(registry, host=config["host"], port=config["port"],
                               token=config["token"]))


if __name__ == "__main__":
    main()
