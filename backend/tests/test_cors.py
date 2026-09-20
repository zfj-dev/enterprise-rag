"""CORS 配置测试：默认来源、main.py 的逗号分隔解析、跨域头回显。"""
from app.config import get_settings


def test_cors_origins_default():
    assert get_settings().cors_origins == "*"


def test_cors_origins_parsing():
    # main.py 的 allow_origins 解析逻辑：逗号分隔 + 去空白
    raw = "http://a, http://b,"
    assert [o.strip() for o in raw.split(",") if o.strip()] == ["http://a", "http://b"]


def test_cors_middleware_configured():
    from starlette.middleware.cors import CORSMiddleware
    from app.main import app

    mw = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert mw, "应配置 CORSMiddleware"
    expected = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
    assert mw[0].kwargs["allow_origins"] == expected


def test_cors_reflects_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    aao = r.headers.get("access-control-allow-origin")
    # allow_origins=* 且**关了** allow_credentials 时，Starlette 输出字面 `*`。
    assert aao == "*"


def test_cors_does_not_echo_arbitrary_origin_with_credentials(client):
    """不许回显任意 Origin + Allow-Credentials（安全审查 H3）。

    本应用鉴权走 `Authorization: Bearer`（令牌在 localStorage），一个 Cookie 都不用它 ——
    所以 `allow_credentials` 本就不需要。而开着它的副作用是：Starlette 在 allow_origins=["*"]
    时会**回显调用方自己的 Origin** 并附上 `Allow-Credentials: true`，
    等于把浏览器这道边界整个让掉。这条断言把「不许再开回去」钉住。
    """
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") == "*"
    assert r.headers.get("access-control-allow-credentials") is None


def test_cors_middleware_credentials_disabled():
    """中间件参数层面也钉一道：allow_credentials 必须是 False。"""
    from starlette.middleware.cors import CORSMiddleware
    from app.main import app

    mw = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert mw and mw[0].kwargs.get("allow_credentials") is False
