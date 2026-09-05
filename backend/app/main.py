"""FastAPI 应用入口：建表、seed 管理员、启动时重建检索索引、挂路由与静态前端、CORS(局域网)。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.v1 import auth, chat, debug, documents, feedback, knowledge, metrics
from app.config import get_settings
from app.db.session import SessionLocal, engine
from app.models.entities import Base, User
from app.utils.security import hash_password

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



@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _seed_admin()
    _reindex()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(knowledge.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(feedback.router, prefix=settings.api_prefix)
    app.include_router(debug.router, prefix=settings.api_prefix)
    app.include_router(metrics.router, prefix=settings.api_prefix)

    # 静态前端（前台直接托管，无需构建即可本地/LAN 使用）
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/health")
    def health():
        return {"status": "ok", "app": settings.app_name, "use_real": settings.use_real}

    @app.middleware("http")
    async def catch_unhandled(request, call_next):
        try:
            return await call_next(request)
        except Exception:
            _error_logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})

    @app.post("/api/v1/client-error")
    async def client_error(payload: dict):
        """前端 window.onerror 上报 JS 错误；落库到 error.log 以便聚合。"""
        _error_logger.error("CLIENT_JS_ERROR %s", payload.get("msg", "")[:2000])
        return {"ok": True}

    return app


def _seed_admin() -> None:
    db: Session = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(username="admin", password_hash=hash_password("admin123"), role="admin"))
            db.commit()
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
