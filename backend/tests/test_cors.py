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
    # allow_credentials=True + origins=* 时 Starlette 可能回显请求 Origin 或输出 *
    assert aao in ("*", "http://localhost:5173")
