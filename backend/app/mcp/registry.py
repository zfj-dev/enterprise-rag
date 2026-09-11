"""按本次请求的上下文造工具集（票 12）。

**范围参数一律服务端注入** —— kb_id / owner_id 由闭包带进来，模型传什么都不看。
这与「检索时权限下推」是同一条纪律：越权不是靠校验参数，而是压根没有那个入口。
"""
from __future__ import annotations

from typing import Any

from app.core.tools import CALCULATOR, Tool, ToolError, ToolRegistry

DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def _top_k(raw: Any) -> int:
    """取 top_k：非法值回落默认，超过上限压到上限 —— 模型说什么都不该把量取爆。"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(1, min(n, MAX_TOP_K))


def kb_retrieve_tool(db, rt, kb_id: str, owner_id: str) -> Tool:
    """绑定了范围的知识库检索工具。检索与普通问答同质（同一个 retrieve_candidates）。"""

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("缺少字符串参数 query")
        from app.services.chat_service import retrieve_candidates, to_sources

        # kb_id / owner_id 取自闭包 —— arguments 里就算塞了范围参数也一律不看
        candidates = retrieve_candidates(db, rt, kb_id=kb_id, owner_id=owner_id,
                                         question=query.strip())
        # 用**问答那一份**映射：这样工具给的 text/页码/分数与 sources 完全同质
        return {"sources": to_sources(candidates[: _top_k(arguments.get("top_k"))])}

    return Tool(
        name="KbRetrieve",
        description="在**当前用户自己的**知识库里检索相关片段，返回带 chunk_id 与页码的来源。"
                    "问文档里写了什么就用它；检索范围由服务端决定，调用方传什么范围参数都没用。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的问题或关键词"},
                "top_k": {"type": "integer",
                          "description": "最多返回几段（默认 %d，上限 %d）" % (DEFAULT_TOP_K, MAX_TOP_K)},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def build_registry(db, rt, user, kb_id: str) -> ToolRegistry:
    """按本次请求的上下文造工具集。票 13 会再挂上 SqlQuery。"""
    return ToolRegistry(tools=[CALCULATOR, kb_retrieve_tool(db, rt, kb_id, user.id)])
