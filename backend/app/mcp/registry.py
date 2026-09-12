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




# ---------- SqlQuery：结构化查询描述，**不接受原始 SQL** ----------
# 注入面按构造消除：能查什么由白名单定死，范围由服务端注入 —— 不是靠转义或黑名单。

MAX_ROWS = 100
_DEFAULT_ROWS = 20
# 过滤方式与聚合函数都用**显式派发表**（与 tools.py 的 _BINOPS 同一风格）。
# 这样加算子漏了实现会当场 KeyError，而不是悄悄退化成别的语义。
_CONDS = {
    "eq": lambda c, v: c == v,
    "ne": lambda c, v: c != v,
    "gt": lambda c, v: c > v,
    "ge": lambda c, v: c >= v,
    "lt": lambda c, v: c < v,
    "le": lambda c, v: c <= v,
    "in": lambda c, v: c.in_(v if isinstance(v, list) else [v]),
    "contains": lambda c, v: c.contains(str(v)),
}
_OPS = tuple(_CONDS)
_AGGS = ("count", "sum", "avg", "min", "max")


def _schema() -> dict:
    """白名单：表 -> {字段名: 列}。凭证类字段压根不在表里，「不可达」是构造出来的。"""
    from app.models.entities import Document, KnowledgeBase

    return {
        "documents": {"filename": Document.filename, "status": Document.status,
                      "page_count": Document.page_count, "chunk_count": Document.chunk_count,
                      "kb_id": Document.kb_id, "created_at": Document.created_at},
        "knowledge_bases": {"name": KnowledgeBase.name, "created_at": KnowledgeBase.created_at},
    }


_NUMERIC = {"documents": ("page_count", "chunk_count"), "knowledge_bases": ()}


def _wanted_rows(raw) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ROWS
    return max(1, min(n, MAX_ROWS))          # 强制上限：模型要多少都不好使


def _condition(col, op: str, value):
    """按算子取值条件。op 已在 _parse_query 里卡过白名单。"""
    return _CONDS[op](col, value)


def _apply_order(q, cols: dict, order: dict | None):
    """排序只在这里做 —— 聚合与非聚合两条路径共用，免得写两遍写歪。"""
    if not order:
        return q
    col = cols[order["field"]]
    return q.order_by(col.desc() if order.get("desc") else col.asc())


def _parse_query(arguments: dict) -> dict:
    """把模型给的结构化描述校一遍。

    越界（表/字段/算子/聚合/排序不在白名单）一律抛 ToolError，不静默纠正；
    只有 `limit` 与省略的 `fields` 是**有意的宽容**：limit 非法回落默认并夹到上限，
    fields 省略则取全表 —— 这两个都在工具的 input_schema 与描述里写明了。
    """
    if not isinstance(arguments, dict):
        raise ToolError("参数必须是一个对象")
    schema = _schema()
    table = arguments.get("table")
    if table not in schema:
        raise ToolError("只能查这几张表：%s" % " / ".join(sorted(schema)))
    cols = schema[table]

    fields = arguments.get("fields") or list(cols)
    if not isinstance(fields, list) or not fields:
        raise ToolError("fields 必须是非空数组")
    for f in fields:
        if f not in cols:
            raise ToolError("表 %s 没有字段 %s（可用：%s）" % (table, f, " / ".join(sorted(cols))))

    parsed_filters = []
    for flt in arguments.get("filters") or []:
        if not isinstance(flt, dict):
            raise ToolError("filters 里每一项都得是对象")
        name, op = flt.get("field"), flt.get("op", "eq")
        if name not in cols:
            raise ToolError("过滤字段不在白名单：%s" % name)
        if op not in _OPS:
            raise ToolError("不支持的过滤方式 %s（可用：%s）" % (op, " / ".join(_OPS)))
        parsed_filters.append((name, op, flt.get("value")))

    agg = arguments.get("aggregate")
    if agg is not None:
        if not isinstance(agg, dict):
            raise ToolError("aggregate 必须是对象")
        fn = agg.get("fn")
        if fn not in _AGGS:
            raise ToolError("不支持的聚合 %s（可用：%s）" % (fn, " / ".join(_AGGS)))
        if fn != "count":
            af = agg.get("field")
            if af not in cols:
                raise ToolError("聚合字段不在白名单：%s" % af)
            if af not in _NUMERIC.get(table, ()):
                raise ToolError("字段 %s 不是数值，不能做 %s" % (af, fn))

    group_by = arguments.get("group_by") or []
    if not isinstance(group_by, list):
        raise ToolError("group_by 必须是数组")
    for f in group_by:
        if f not in cols:
            raise ToolError("分组字段不在白名单：%s" % f)

    order = arguments.get("order_by")
    if order is not None:
        if not isinstance(order, dict) or order.get("field") not in cols:
            raise ToolError("order_by.field 不在白名单")

    return {"table": table, "fields": fields, "filters": parsed_filters, "aggregate": agg,
            "group_by": group_by, "order_by": order, "limit": _wanted_rows(arguments.get("limit"))}


def _run_query(db, owner_id: str, spec: dict) -> dict:
    """只读查询。属主范围由这里注入，模型传什么范围参数都没用。"""
    from sqlalchemy import func

    from app.models.entities import Document, KnowledgeBase

    cols = _schema()[spec["table"]]
    model = Document if spec["table"] == "documents" else KnowledgeBase

    def fresh():
        q = db.query(model).filter(model.owner_id == owner_id)      # ← 范围服务端注入
        for name, op, value in spec["filters"]:
            q = q.filter(_condition(cols[name], op, value))
        return q

    agg = spec["aggregate"]
    if agg is not None:
        fn, field = agg["fn"], agg.get("field")
        expr = func.count() if fn == "count" else getattr(func, fn)(cols[field])
        label = "count" if fn == "count" else "%s_%s" % (fn, field)
        if spec["group_by"]:
            groups = [cols[f] for f in spec["group_by"]]
            q = fresh().with_entities(*groups, expr.label(label)).group_by(*groups)
            # 没点名排序时按聚合值倒序（"哪个库文档最多"用得着）
            q = _apply_order(q, cols, spec["order_by"]) if spec["order_by"] else q.order_by(expr.desc())
            rows = q.limit(spec["limit"]).all()
            out = [dict(zip(spec["group_by"], r[:len(groups)]), **{label: r[-1]}) for r in rows]
        else:
            out = [{label: fresh().with_entities(expr).scalar()}]
        return {"rows": out, "count": len(out), "table": spec["table"]}

    q = _apply_order(fresh().with_entities(*[cols[f] for f in spec["fields"]]),
                     cols, spec["order_by"])
    rows = q.limit(spec["limit"]).all()
    return {"rows": [dict(zip(spec["fields"], r)) for r in rows],
            "count": len(rows), "table": spec["table"]}


def sql_query_tool(db, user) -> Tool:
    """文档 / 知识库**元数据**查询。只吃结构化描述，**绝不接受原始 SQL 文本**。"""

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return _run_query(db, user.id, _parse_query(arguments))

    return Tool(
        name="SqlQuery",
        description="查**文档与知识库的元数据**（文件名 / 状态 / 页数 / 分块数 / 所属库 / 时间）。"
                    "要传结构化描述（table / fields / filters / aggregate / group_by / "
                    "order_by / limit），**不是 SQL 文本**。范围由服务端决定，只读、有行数上限。",
        input_schema={
            "type": "object",
            "required": ["table"],
            "properties": {
                "table": {"type": "string", "enum": ["documents", "knowledge_bases"]},
                "fields": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"field": {"type": "string"},
                                   "op": {"type": "string", "enum": list(_OPS)},
                                   "value": {}},
                    "required": ["field"]}},
                "aggregate": {"type": "object",
                              "properties": {"fn": {"type": "string", "enum": list(_AGGS)},
                                             "field": {"type": "string"}}},
                "group_by": {"type": "array", "items": {"type": "string"}},
                "order_by": {"type": "object",
                             "properties": {"field": {"type": "string"},
                                            "desc": {"type": "boolean"}}},
                "limit": {"type": "integer",
                          "description": "最多几行（默认 %d，上限 %d）" % (_DEFAULT_ROWS, MAX_ROWS)},
            },
            "additionalProperties": False,
        },
        handler=handler,
    )


def build_registry(db, rt, user, kb_id: str) -> ToolRegistry:
    """按本次请求的上下文造工具集。"""
    return ToolRegistry(tools=[CALCULATOR,
                               kb_retrieve_tool(db, rt, kb_id, user.id),
                               sql_query_tool(db, user)])
