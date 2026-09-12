"""按**显式配置**决定这次挂载的身份与范围（票 16）。

外部 MCP 客户端（Claude Desktop / Claude Code 等）以子进程方式起这个 server —— 它没有
HTTP 请求上下文。所以 owner_id / kb_id 只能来自**操作者的配置**（--user / --kb），
**绝不来自客户端传的参数** —— 与「检索时范围服务端注入」是同一条纪律。
"""
from __future__ import annotations

import sys

from app.core.tools import ToolRegistry, default_registry


def registry_for_mount(username: str | None, kb_id: str | None) -> ToolRegistry:
    """配了身份就挂上需要上下文的工具（KbRetrieve / SqlQuery），没配就只挂 Calculator。

    身份或库对不上（用户不存在 / 库不属于该用户 / 只给库不给用户）一律 ValueError ——
    挂载点宁可起不来，也不能静默降级成「看着挂了三个、其实查别人的库」。
    """
    if kb_id and not username:
        raise ValueError("给了 --kb 却没给 --user：身份不明，不能只按库挂载")
    if not username:
        return default_registry()

    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.mcp.registry import build_registry
    from app.models.entities import KnowledgeBase, User
    from app.services.document_service import reindex_all

    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise ValueError("没有这个用户：%s（--user 得是本机库里的用户名）" % username)

    kb = None
    if kb_id:
        kb = db.get(KnowledgeBase, kb_id)
        if kb is None:
            raise ValueError("没有这个知识库：%s" % kb_id)
        if kb.owner_id != user.id:
            raise ValueError("知识库 %s 不属于用户 %s —— 挂载范围不能跨人" % (kb_id, username))
    else:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.owner_id == user.id).first()
        if kb is None:
            raise ValueError("用户 %s 名下没有任何知识库，挂不了检索类工具" % username)

    # 常驻进程的内存索引一开始是空的：从库里已入库的分块重建，KbRetrieve 才查得到
    rt = build_runtime()
    reindex_all(db, rt)
    # 会话要活到进程结束（工具的闭包靠它查库）；但 reindex 那个读事务没必要一直攥着 ——
    # sqlite 下常开事务会拿着读锁，把主应用那边的写顶成 database is locked
    print("[mcp] 身份=%s 库=%s(%s)" % (user.username, kb.id, kb.name),
          file=sys.stderr, flush=True)
    db.rollback()
    return build_registry(db, rt, user, kb.id)
