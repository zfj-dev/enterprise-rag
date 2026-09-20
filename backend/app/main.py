"""FastAPI 应用入口：建表、seed 管理员、启动时重建检索索引、挂路由与静态前端、CORS(局域网)。"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.v1 import (auth, chat, debug, documents, feedback, knowledge, llm, memory,
                        metrics)
from app.config import DEMO_ADMIN_PASSWORD, get_settings
from app.db.session import SessionLocal, engine
from app.models.entities import Base, User
from app.utils.security import hash_password, verify_password
from app.utils.ratelimit import SlidingWindowLimiter

import logging
import logging.handlers

settings = get_settings()

# ---------- 全局异常日志（按天轮转，写 logs/error.log） ----------
_error_logger = logging.getLogger("app.error")
_error_logger.setLevel(logging.ERROR)
os.makedirs("logs", exist_ok=True)
_h = logging.handlers.TimedRotatingFileHandler("logs/error.log", when="midnight", backupCount=7, encoding="utf-8")
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_error_logger.addHandler(_h)
_error_logger.propagate = False

# ---------- 根 logger：让各模块 logging.getLogger(__name__).warning 落到 stderr（回退/吞异常可见），格式统一 ----------
_root = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in _root.handlers):
    _root.addHandler(logging.StreamHandler())
_root.setLevel(logging.WARNING)



# ---------- /client-error 的轻量限流（按客户端 IP） ----------
# 键只能是 IP：这个端点**必须**未鉴权（登录页自己的 JS 报错也要能上来），没法用用户身份做键。
# ⚠️ 走反代（deploy/Caddyfile）时 `request.client.host` 是**反代的地址** —— 所有客户端共用
# 一个桶（30/分钟）。方向是「限得更严」而非放宽，不构成漏洞；真按 IP 限流需配可信代理头。
_CE_WINDOW = 60.0
_CE_MAX = 30
_CE_MAX_KEYS = 4096
_client_error_limiter = SlidingWindowLimiter(_CE_WINDOW, max_keys=_CE_MAX_KEYS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    from app.db.migrate import ensure_sqlite_columns

    ensure_sqlite_columns(engine)     # 已有库补上新增列（create_all 只会建表）
    _seed_admin()
    _recover_stale_documents()
    _reindex()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        """校验失败时**不回显请求体**。

        FastAPI 默认会把 `input`（也就是整个 body）塞进 422 详情里 —— 而 BYOK 的 body 里带着
        明文 key（票 32 的硬要求：任何响应都不含明文 key）。这里只留定位与原因。
        """
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]})

    _origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if "*" in _origins and settings.use_real:
        logging.getLogger(__name__).warning(
            "真实模式仍用 CORS_ORIGINS=*：任意网站都能调用本 API。"
            "鉴权走 Authorization 头（浏览器不会自动带上），所以不等于 CSRF，"
            "但生产建议显式列出前端来源，例如 CORS_ORIGINS=http://rag.example.com")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        # **关掉凭证**：本应用的鉴权走 `Authorization: Bearer`（localStorage 里的令牌），
        # 一个 Cookie 都不用。开着它的副作用是：Starlette 在 allow_origins=["*"] 时会
        # **回显任意 Origin** 并附上 Allow-Credentials，等于把浏览器这道边界整个让掉
        # （安全审查 H3，已在本仓库实测）。
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(knowledge.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(feedback.router, prefix=settings.api_prefix)
    app.include_router(debug.router, prefix=settings.api_prefix)
    app.include_router(memory.router, prefix=settings.api_prefix)
    app.include_router(metrics.router, prefix=settings.api_prefix)
    app.include_router(llm.router, prefix=settings.api_prefix)

    # 静态前端（前台直接托管，无需构建即可本地/LAN 使用）
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/health")
    def health():
        # agent_enabled 一并发给前端：代理按钮该不该出现由服务端说了算，前端别自己猜（票 37）
        return {"status": "ok", "app": settings.app_name, "use_real": settings.use_real,
                "agent_enabled": settings.agent_enabled}

    @app.middleware("http")
    async def catch_unhandled(request, call_next):
        try:
            return await call_next(request)
        except Exception:
            _error_logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})

    @app.post("/api/v1/client-error")
    async def client_error(payload: dict, request: Request):
        """前端 window.onerror 上报 JS 错误；落进 error.log 以便聚合。

        **故意不要求鉴权**：登录页自己的报错也得能上来。代价是这个端点是公开的，
        所以补两道（安全审查 H2）：
        - 按 IP 限流 —— 否则就是一条把磁盘写满的路径；
        - 剥掉 `\\r`/`\\n` —— 否则上报内容能伪造出整行日志（日志注入）。
        """
        ip = (request.client.host if request.client else "") or "?"
        if not _client_error_limiter.allow(ip, _CE_MAX):
            return JSONResponse(status_code=429, content={"ok": False, "reason": "上报过于频繁"})
        msg = str(payload.get("msg", ""))[:2000]
        _error_logger.error("CLIENT_JS_ERROR ip=%s %s",
                            ip, msg.replace("\r", " ").replace("\n", " "))
        return {"ok": True}

    return app


def _seed_admin() -> None:
    """首次启动建一个管理员。**口令不写死**。

    安全审查 B1：原来无条件用 `admin123`，而这个口令写在 README 与 CLAUDE.md 里 ——
    等于把管理员身份公开。现在分三种情况：
    - 配了 `ADMIN_PASSWORD` → 用它（真实模式要求至少 12 位，太短直接拒绝启动）；
    - 真实模式没配 → **随机生成并打印一次**（宁可让操作者去日志里抄一次，
      也不能让一个全网都知道的口令留在线上）；
    - 演示模式没配 → 保留 `admin123`，但**每次都提醒**这是公开口令。

    另外：库里已存在 admin 时不动它（口令轮换走 `POST /auth/change-password`），
    但会检查它是不是还挂着那个公开口令，是就喊一声。
    """
    s = get_settings()
    # 读**配置字段**而不是 os.environ：`.env` 里的值 pydantic 只灌进 Settings，
    # 裸读环境变量会让「按 .env.example 配好」的人静默回落公开口令。
    pwd = (s.admin_password or "").strip()
    if pwd:
        if s.use_real and len(pwd) < 12:
            raise RuntimeError("ADMIN_PASSWORD 太短（真实模式至少 12 字符）")
    elif s.use_real:
        pwd = secrets.token_urlsafe(12)
        print(f"[seed] 管理员 admin 的初始口令（**仅本次打印**，登录后请立即修改）：{pwd}")
    else:
        pwd = DEMO_ADMIN_PASSWORD
        print(f"[seed] 注意：演示模式 admin 的口令是公开的 {DEMO_ADMIN_PASSWORD}。"
              "上线前请设 ADMIN_PASSWORD，或登录后用 /auth/change-password 改掉。")

    db: Session = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if admin is None:
            db.add(User(username="admin", password_hash=hash_password(pwd), role="admin"))
            db.commit()
        elif verify_password(DEMO_ADMIN_PASSWORD, admin.password_hash):
            # 口令是公开的这件事，光看数据库看不出来 —— 主动验一次并在日志里说清楚，
            # 否则「已经建成 admin」这件事会让人以为「口令的问题已经处理过了」。
            print(f"[seed] 注意：管理员 admin 的口令仍是公开的 {DEMO_ADMIN_PASSWORD} —— "
                  "请立刻用 POST /api/v1/auth/change-password 改掉。")
    finally:
        db.close()


def _recover_stale_documents() -> None:
    """把上一次进程留下的 `processing` 标成 failed —— 后台线程没了，没人会再来收尾。"""
    db: Session = SessionLocal()
    try:
        from app.services.document_service import fail_stale_processing

        fail_stale_processing(db)
    except Exception as e:  # noqa
        print(f"[recover] 跳过（{e}）")
    finally:
        db.close()


def _reindex() -> None:
    """启动时从数据库重建内存向量库/BM25 索引（否则重启后检索为空）。"""
    db: Session = SessionLocal()
    try:
        from app.api.deps import get_runtime
        from app.services.document_service import reindex_all

        n = reindex_all(db, get_runtime())
        if n:
            print(f"[reindex] 已从数据库重建 {n} 个分块的索引")
    except Exception as e:  # noqa
        print(f"[reindex] 跳过（{e}）")
    finally:
        db.close()


app = create_app()
