"""MCP server 对外可挂载（票 16 / #23）：三件套可发现、范围按**显式配置**注入、默认仅本地。

外部客户端以子进程起这个 server，没有 HTTP 请求上下文 —— 身份与范围只能来自操作者的
显式配置（--user / --kb），绝不来自客户端传的参数。这里就测这条纪律。
"""
from __future__ import annotations

import pytest

from app.mcp.mount import registry_for_mount
from app.mcp.server import build_http_app, resolve_transport
from tests.helpers import register_and_kb, wait_until

DOC = "比亚迪2025年营业收入为803.96亿元。"


def _seed(client, name: str):
    """建一个带文档的库，返回 (username, kb_id, doc_id)。"""
    from app.db.session import SessionLocal
    from app.models.entities import Document

    H, uid, kb = register_and_kb(client, name)
    up = client.post("/api/v1/documents?kb_id=%s" % kb, headers=H,
                     files={"file": ("annual.txt", DOC, "text/plain")}).json()
    assert wait_until(lambda: client.get("/api/v1/documents/%s" % up["id"], headers=H)
                      .json().get("status") in ("indexed", "failed")), "文档未入库"
    db = SessionLocal()
    try:
        doc_id = db.query(Document).filter(Document.kb_id == kb).first().id
    finally:
        db.close()
    return name, kb, doc_id


def _names(registry) -> set:
    return {s["name"] for s in registry.specs()}


# ---------- 可发现性 ----------

def test_without_a_configured_identity_only_context_free_tools_are_mounted():
    """没配身份就没有范围可用 —— 只挂不依赖上下文的工具，不假装能给谁查库。"""
    assert _names(registry_for_mount(None, None)) == {"Calculator"}


def test_all_three_tools_are_discoverable_with_names_descriptions_and_schemas(client):
    """三个工具都要能被「发现」：名字 + 描述 + 输入 schema（客户端据此决定怎么调）。"""
    name, kb, _ = _seed(client, "mcp_discover")
    specs = {s["name"]: s for s in registry_for_mount(name, kb).specs()}

    assert set(specs) == {"KbRetrieve", "SqlQuery", "Calculator"}
    for s in specs.values():
        assert (s.get("description") or "").strip()
        schema = s.get("inputSchema") or {}
        assert schema.get("type") == "object"
    assert "query" in specs["KbRetrieve"]["inputSchema"]["required"]
    assert specs["SqlQuery"]["inputSchema"]["properties"]["table"]["enum"] == [
        "documents", "knowledge_bases"]


def test_the_configured_scope_wins_over_client_supplied_arguments(client):
    """范围由挂载配置注入：客户端塞 kb_id / owner_id 一律不看（越权拿不到别人的块）。"""
    name_a, kb_a, doc_a = _seed(client, "mcp_scope_a")
    name_b, kb_b, doc_b = _seed(client, "mcp_scope_b")

    registry = registry_for_mount(name_a, kb_a)
    got = registry.call("KbRetrieve", {"query": "营业收入",
                                       "kb_id": kb_b, "owner_id": "someone-else"})

    assert got["sources"], "配置范围没查到东西"
    assert {s["doc_id"] for s in got["sources"]} == {doc_a}   # 全是自己的


def test_an_unknown_user_is_refused(client):
    """配了个不存在的身份 -> 当场拒绝启动，而不是静默降级成「谁都能查」。"""
    with pytest.raises(ValueError) as e:
        registry_for_mount("no-such-user", None)
    assert "no-such-user" in str(e.value)


def test_a_kb_that_belongs_to_someone_else_is_refused(client):
    """不能拿 A 的身份挂 B 的库。"""
    name_a, _, _ = _seed(client, "mcp_owner_a")
    _, kb_b, _ = _seed(client, "mcp_owner_b")

    with pytest.raises(ValueError) as e:
        registry_for_mount(name_a, kb_b)
    assert "不属于" in str(e.value) or "not" in str(e.value).lower()


# ---------- 传输与鉴权 ----------

def test_stdio_is_the_default_transport():
    """默认本地 stdio：不占端口、不出网络。"""
    assert resolve_transport({}) == {"transport": "stdio"}


def test_network_transport_needs_an_explicit_allow_and_a_token():
    """对外暴露要显式两件套：允许联网 + 令牌。缺一个都不启动。"""
    with pytest.raises(ValueError) as e:
        resolve_transport({"MCP_TRANSPORT": "http", "MCP_TOKEN": "t"})
    assert "MCP_ALLOW_NETWORK" in str(e.value)

    with pytest.raises(ValueError) as e:
        resolve_transport({"MCP_TRANSPORT": "http", "MCP_ALLOW_NETWORK": "true"})
    assert "MCP_TOKEN" in str(e.value)

    got = resolve_transport({"MCP_TRANSPORT": "http", "MCP_ALLOW_NETWORK": "true",
                             "MCP_TOKEN": "secret-token"})
    assert got["transport"] == "http"
    assert got["host"] == "127.0.0.1"        # 就算开了网络传输，也默认只绑本机
    assert got["token"] == "secret-token"


def test_exposing_beyond_localhost_also_needs_the_explicit_allow():
    """MCP_HOST 想绑到 0.0.0.0 也得走允许联网那条路 —— 不然等于悄悄开了外网。"""
    with pytest.raises(ValueError):
        resolve_transport({"MCP_TRANSPORT": "stdio", "MCP_HOST": "0.0.0.0"})


def test_the_http_app_is_guarded_by_the_bearer_token(client):
    """挂出去的那份 app 真的带鉴权：令牌不对 401；对了才放行进 MCP 层。

    base_url 必须带端口：SDK 对本机 host 会开 DNS-rebinding 保护、白名单形如
    `127.0.0.1:*`，用 TestClient 默认的 `testserver` 会先吃 421 —— 那样断言的
    就不再是「鉴权放行」了（审查抓到过这个假绿）。
    """
    pytest.importorskip("mcp.types", reason="需要 mcp SDK（见 requirements-real.txt）")
    from starlette.testclient import TestClient

    name, kb, _ = _seed(client, "mcp_http")
    app = build_http_app(registry_for_mount(name, kb), host="127.0.0.1", token="secret-token")

    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert c.post("/mcp", json=payload).status_code == 401
        assert c.post("/mcp", json=payload,
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.post("/mcp", json=payload,
                    headers={"Authorization": "Bearer secret-token",
                             "Accept": "application/json, text/event-stream"})
        assert "Missing session ID" in ok.text    # 真的进了 MCP 层（还没握手，所以没有会话）


def test_a_kb_without_a_user_is_refused(client):
    """只给库不给身份 -> 拒绝：不然「以谁的名义查」是笔糊涂账。"""
    _, kb, _ = _seed(client, "mcp_kb_only")

    with pytest.raises(ValueError) as e:
        registry_for_mount(None, kb)
    assert "--user" in str(e.value)


def test_an_unknown_transport_is_refused():
    """只认自己真实现的传输 —— 收下 sse 却发 http 就是骗配置的人。"""
    with pytest.raises(ValueError) as e:
        resolve_transport({"MCP_TRANSPORT": "sse", "MCP_ALLOW_NETWORK": "true",
                           "MCP_TOKEN": "t"})
    assert "sse" in str(e.value)


def test_an_invalid_port_is_refused():
    with pytest.raises(ValueError):
        resolve_transport({"MCP_TRANSPORT": "http", "MCP_ALLOW_NETWORK": "true",
                           "MCP_TOKEN": "t", "MCP_PORT": "abc"})
    with pytest.raises(ValueError):
        resolve_transport({"MCP_TRANSPORT": "http", "MCP_ALLOW_NETWORK": "true",
                           "MCP_TOKEN": "t", "MCP_PORT": "70000"})
