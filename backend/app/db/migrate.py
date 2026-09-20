"""已有库的补列（本仓库没有迁移框架，`create_all` 只建表、不会给旧表加列）。

加列本身是幂等的、也不需要回填（默认值就是语义），所以这里只做「缺了就补」这一件事，
并把补了哪些列返回给启动日志 —— 免得升级后第一句话撞上 no such column 才知道。

**名字不再带 `sqlite`**（原来叫 `ensure_sqlite_columns`）：这里用的 DDL 与判断
（`ALTER TABLE x ADD COLUMN y` + `inspect()`）是引擎中立的，对 Postgres 同样成立 ——
名字写着 sqlite 会让人误以为生产走的是另一条路（安全审查 F14）。

⚠️ 已知边界：只补列，不建表、不改类型、不删列。长期演进还是得上 Alembic；在那之前，
加字段请一律「只加带默认值的列」。（Postgres 这条路径**未经真机验证** —— 仓库的测试
全跑 sqlite。）
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 表 -> {列名: 加列的 DDL 片段}
_NEEDED: dict[str, dict[str, str]] = {
    "chat_sessions": {                       # 票 18：会话上的滚动摘要 + 游标
        "summary": "summary TEXT DEFAULT ''",
        "summary_upto": "summary_upto VARCHAR(32)",
    },
    "usage_records": {                       # 票 28：用量记录上的折算费用与单价口径
        "cost": "cost FLOAT",
        "price_note": "price_note TEXT DEFAULT ''",
    },
}


def ensure_missing_columns(engine) -> list[str]:
    """给已有的库补上缺失的列；返回这次补了哪些（"表.列"）。"""
    from sqlalchemy import inspect, text

    added: list[str] = []
    inspector = inspect(engine)
    for table, columns in _NEEDED.items():
        if not inspector.has_table(table):
            continue                          # 全新库由 create_all 直接建对，不用补
        have = {c["name"] for c in inspector.get_columns(table)}
        with engine.begin() as conn:
            for name, ddl in columns.items():
                if name in have:
                    continue
                conn.execute(text("ALTER TABLE %s ADD COLUMN %s" % (table, ddl)))
                added.append("%s.%s" % (table, name))
    if added:
        logger.warning("已给已有库补上新列：%s", ", ".join(added))
    return added
