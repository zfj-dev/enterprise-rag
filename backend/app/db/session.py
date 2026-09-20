"""SQLAlchemy 引擎与会话（用 database_url；sqlite 测试 / postgres 生产通用）。"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _make_engine(url: str, echo: bool = False):
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _enable_sqlite_foreign_keys(engine)
    return engine


def _enable_sqlite_foreign_keys(engine) -> None:
    """让 sqlite **真的校验外键**。

    sqlite 出于历史兼容**默认不校验外键**，而生产用的 Postgres 会 —— 两边语义不一致，
    于是「测试全绿、上线 500」这类问题会一直藏着（安全审查 F9：删会话时
    `feedback.message_id` 的外键就让 Postgres 直接拒了，sqlite 却装作没事）。
    打开之后，本地跑出来的行为才和线上一致。
    """
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):        # noqa: ANN001 —— DBAPI 连接，没有类型可标
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()


_settings = get_settings()
engine = _make_engine(_settings.database_url, _settings.db_echo)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
