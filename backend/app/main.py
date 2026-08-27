"""FastAPI 应用入口：建表、seed 管理员、挂载路由与静态前端、CORS(局域网)。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.v1 import auth, chat, debug, documents, feedback, knowledge, metrics
from app.config import get_settings
from app.db.session import SessionLocal, engine
from app.models.entities import Base, User
from app.utils.security import hash_password

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _seed_admin()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 局域网 demo；生产应收紧
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

    # 静态前端（index.html；无需构建即可本地/LAN 使用）
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/health")
    def health():
        return {"status": "ok", "app": settings.app_name, "use_real": settings.use_real}

    return app


def _seed_admin() -> None:
    db: Session = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(username="admin", password_hash=hash_password("admin123"), role="admin"))
            db.commit()
    finally:
        db.close()


app = create_app()
