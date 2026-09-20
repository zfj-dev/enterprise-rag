"""FastAPI 应用入口：建表、seed 管理员、启动时重建检索索引、挂路由与静态前端、CORS(局域网)。"""
from __future__ import annotations

import os
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
from app.core.schemas import MIN_PASSWORD_CHARS, ClientErrorIn
from app.utils.bodylimit import BodySizeLimitMiddleware
from app.utils.ratelimit import SlidingWindowLimiter
from app.utils.security import hash_password, verify_password

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
    from app.db.migrate import ensure_missing_columns

    ensure_missing_columns(engine)     # 已有库补上新增列（create_all 只会建表）
    _seed_admin()
    _recover_stale_documents()
    _reindex()
    yield
    _drain_indexing()


def _drain_indexing() -> None:
    """优雅关停：给在跑的（异步）索引线程一点时间收尾。

    不等的话，正在索引的文档会被腰斩在「写了一半」的状态；下次启动
    `fail_stale_processing` 会标成 failed 让用户重传（兜底还在），但用户白传一次。
    """
    from app.services import document_service

    left = document_service.wait_for_processing()
    if left:
        print(f"[shutdown] {left} 个索引线程未在 "
              f"{document_service.SHUTDOWN_WAIT_SECONDS:.0f}s 内收尾 —— "
              "下次启动会把它们标成失败，请让用户重传")


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

    # 请求体封顶。**加在 CORS 之前 = 更内层**（Starlette 的 add_middleware 是前插，
    # 最后加的走在最前面）—— 这样它拒掉的 413 会经过 CORS 从而带上跨域响应头；
    # 反过来放最外层的话，跨域前端会把 413 看成「CORS 失败」。它仍在路由读 body 之前，
    # 所以该拦的照样拦得住。上传天然大得多，按路径前缀单独放宽 —— 但也不是无上限。
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_body_mb * 1024 * 1024,
        overrides={settings.api_prefix + "/documents":
                   (settings.max_upload_mb + 5) * 1024 * 1024},
    )

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
        """健康检查 + 前端启动时要的两个开关。

        **这两个字段故意不鉴权**：前端在**登录之前**就要按它们渲染 —— 模式徽章看
        `use_real`、「深度思考」按钮该不该出现看 `agent_enabled`（票 37，由服务端说了算，
        前端别自己猜）。安全审查 F13 建议把它们挪到鉴权接口后面，但那样会把这两处 UI
        打回「自己猜」。两个值的敏感度也低：只是「是不是演示模式」和「代理链路开没开」。
        """
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
    async def client_error(payload: ClientErrorIn, request: Request):
        """前端 window.onerror 上报 JS 错误；落进 error.log 以便聚合。

        **故意不要求鉴权**：登录页自己的报错也得能上来。代价是这个端点是公开的，
        所以补三道（安全审查 H2 / F1）：
        - 按 IP 限流 —— 否则就是一条把磁盘写满的路径；
        - 剥掉 `\\r`/`\\n` —— 否则上报内容能伪造出整行日志（日志注入）；
        - 字段与长度由 `ClientErrorIn` 卡死，请求体本身再由 BodySizeLimitMiddleware 封顶。
        """
        ip = (request.client.host if request.client else "") or "?"
        if not _client_error_limiter.allow(ip, _CE_MAX):
            return JSONResponse(status_code=429, content={"ok": False, "reason": "上报过于频繁"})
        _error_logger.error("CLIENT_JS_ERROR ip=%s %s",
                            ip, payload.msg.replace("\r", " ").replace("\n", " "))
        return {"ok": True}

    return app


def _seed_admin() -> None:
    """首次启动建一个管理员。**口令不写死、也不打进日志**。

    安全审查 B1 + F11：
    - 配了 `ADMIN_PASSWORD` → 用它（真实模式要求至少 12 位，太短拒绝启动）；
    - 真实模式没配、且**确实要建号** → 拒绝启动。不生成、不打印：容器日志往往会被采集，
      把初始口令写进去等于换个地方泄漏；而真实模式本来也没有「合理的默认口令」可言；
    - 演示模式没配 → 保留公开的 `admin123`，但**每次启动都提醒**。

    库里已存在 admin 时**不会**重建（口令轮换走 `POST /auth/change-password`）——
    所以这条规则不会让既有部署起不来；但会验一下它是不是还挂着那个公开口令，是就喊一声。
    """
    s = get_settings()
    # 读**配置字段**而不是 os.environ：`.env` 里的值 pydantic 只灌进 Settings，
    # 裸读环境变量会让「按 .env.example 配好」的人静默回落公开口令。
    pwd = (s.admin_password or "").strip()

    db: Session = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if admin is not None:
            # 口令是公开的这件事，光看数据库看不出来 —— 主动验一次并在日志里说清楚，
            # 否则「已经建成 admin」这件事会让人以为「口令的问题已经处理过了」。
            if verify_password(DEMO_ADMIN_PASSWORD, admin.password_hash):
                print(f"[seed] 注意：管理员 admin 的口令仍是公开的 {DEMO_ADMIN_PASSWORD} —— "
                      "请立刻用 POST /api/v1/auth/change-password 改掉。")
            return

        # 口令强度只在**确实要建号**时校验：库里已有 admin 的既有部署不该因为 .env 里
        # 那个值（压根不会被用到）而拒绝启动。
        if pwd and s.use_real and len(pwd) < MIN_PASSWORD_CHARS:
            raise RuntimeError("ADMIN_PASSWORD 太短（真实模式至少 %d 字符）" % MIN_PASSWORD_CHARS)

        if not pwd:
            if s.use_real:
                raise RuntimeError(
                    "首次启动要建管理员，但没配 ADMIN_PASSWORD —— 真实模式不接受默认口令，"
                    "也不会把生成的口令打进日志（那是换个地方泄漏）。"
                    "请设 ADMIN_PASSWORD（≥%d 字符）后重启。" % MIN_PASSWORD_CHARS)
            pwd = DEMO_ADMIN_PASSWORD
            print(f"[seed] 注意：演示模式 admin 的口令是公开的 {DEMO_ADMIN_PASSWORD}。"
                  "上线前请设 ADMIN_PASSWORD，或登录后用 /auth/change-password 改掉。")

        db.add(User(username="admin", password_hash=hash_password(pwd), role="admin"))
        db.commit()
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
