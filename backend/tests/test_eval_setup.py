"""评测脚手架：离线脚本要能在一台老库上独立跑起来 —— 建表之外还得补列。"""
from __future__ import annotations


def test_ensure_schema_backfills_columns_too(monkeypatch):
    """`create_all` 只建表、不会给旧表加列；漏了补列这一步，评测在老库上当场撞 no such column。

    （补列本身在 test_chat_summary 里单独验过；这条盯的是**评测脚本有没有接上它**。）
    """
    import app.db.migrate as migrate
    import app.db.session as session
    from app import eval_setup

    seen: list = []
    monkeypatch.setattr(migrate, "ensure_sqlite_columns", lambda engine: seen.append(engine))

    eval_setup.ensure_schema()

    assert seen == [session.engine]      # 建完表就把补列跑在与 app 同一个引擎上
