"""SqlQuery 工具（票 13 / #20）：只吃结构化描述、白名单外一律拒、范围服务端注入。"""
from __future__ import annotations

import pytest

from app.core.tools import ToolError
from tests.helpers import register_and_kb, wait_until

TEXT = "比亚迪安全手册，关于电池和充电的规范说明。"


def _seed(client, name, files=("manual.txt",)):
    H, uid, kb = register_and_kb(client, name)
    for fname in files:
        up = client.post("/api/v1/documents?kb_id=%s" % kb, headers=H,
                         files={"file": (fname, TEXT, "text/plain")}).json()
        assert wait_until(lambda: client.get("/api/v1/documents/%s" % up["id"], headers=H)
                          .json().get("status") in ("indexed", "failed")), "文档未入库"
    return H, uid, kb


def _sql(uid, kb):
    """造工具（范围由服务端注入）。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.mcp.registry import build_registry
    from app.models.entities import User

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uid).first()
        return build_registry(db, get_runtime(), user, kb).get("SqlQuery")
    finally:
        db.close()


# ---------- 正常查询 ----------

def test_lists_my_documents(client):
    _H, uid, kb = _seed(client, "sql1")
    got = _sql(uid, kb).call({"table": "documents", "fields": ["filename", "status"]})

    assert got["table"] == "documents"
    assert got["count"] == 1
    assert got["rows"][0]["filename"] == "manual.txt"
    assert got["rows"][0]["status"] == "indexed"


def test_filter_and_aggregate(client):
    _H, uid, kb = _seed(client, "sql2")
    tool = _sql(uid, kb)

    assert tool.call({"table": "documents", "filters": [{"field": "status", "op": "eq",
                                                         "value": "indexed"}]})["count"] == 1
    assert tool.call({"table": "documents", "filters": [{"field": "status", "op": "eq",
                                                         "value": "failed"}]})["count"] == 0
    got = tool.call({"table": "documents", "aggregate": {"fn": "count"}})
    assert got["rows"][0]["count"] == 1
    got = tool.call({"table": "documents", "aggregate": {"fn": "sum", "field": "chunk_count"}})
    assert got["rows"][0]["sum_chunk_count"] >= 1


def test_group_by_answers_how_many_docs_per_kb(client):
    """「每个知识库有多少文档」——就是这个工具要回答的那类问题。"""
    _H, uid, kb = _seed(client, "sql3", files=("a.txt", "b.txt"))
    got = _sql(uid, kb).call({"table": "documents", "group_by": ["kb_id"],
                              "aggregate": {"fn": "count"}})

    assert got["rows"] == [{"kb_id": kb, "count": 2}]


# ---------- 白名单之外一律拒绝 ----------

@pytest.mark.parametrize("arguments", [
    {"table": "users"},                                        # 表不在白名单（凭证就该查不到）
    {"table": "documents; DROP TABLE users"},                  # 塞原始 SQL 文本
    {"table": "documents", "fields": ["file_path"]},           # 字段不在白名单
    {"table": "documents", "fields": ["owner_id"]},
    {"table": "documents", "filters": [{"field": "password_hash"}]},
    {"table": "documents", "filters": [{"field": "status", "op": "regex", "value": "x"}]},
    {"table": "documents", "aggregate": {"fn": "delete"}},
    {"table": "documents", "aggregate": {"fn": "avg", "field": "filename"}},   # 非数值字段
    {"table": "documents", "aggregate": {"fn": "sum", "field": "owner_id"}},
    {"table": "documents", "group_by": ["password_hash"]},
    {"table": "documents", "order_by": {"field": "not_a_field"}},
    {},                                                        # 连表都没给
])
def test_anything_outside_the_whitelist_is_rejected(client, arguments):
    _H, uid, kb = _seed(client, "sql4")
    with pytest.raises(ToolError):
        _sql(uid, kb).call(arguments)


def test_limit_is_clamped(client):
    _H, uid, kb = _seed(client, "sql5", files=("a.txt", "b.txt", "c.txt"))
    got = _sql(uid, kb).call({"table": "documents", "limit": 1})
    assert len(got["rows"]) == 1

    got = _sql(uid, kb).call({"table": "documents", "limit": "乱写的"})
    assert len(got["rows"]) == 3          # 非法回落默认（20），这里不到 20 就全给


def test_row_count_is_capped_even_if_the_model_asks_for_more(client):
    from app.mcp.registry import MAX_ROWS
    _H, uid, kb = _seed(client, "sql6")
    got = _sql(uid, kb).call({"table": "documents", "limit": 9999})
    assert len(got["rows"]) <= MAX_ROWS


# ---------- 范围服务端注入 ----------

def test_only_my_own_metadata_is_visible(client):
    """两个用户各有一份**同名**文档：只按属主区分，拿回来的必须全是自己的。"""
    _H1, uid1, kb1 = _seed(client, "sqlA")
    _H2, uid2, kb2 = _seed(client, "sqlB")

    rows_a = _sql(uid1, kb1).call({"table": "documents", "fields": ["kb_id"]})["rows"]
    rows_b = _sql(uid2, kb2).call({"table": "documents", "fields": ["kb_id"]})["rows"]

    assert rows_a == [{"kb_id": kb1}]
    assert rows_b == [{"kb_id": kb2}]


def test_scope_parameters_from_the_model_do_not_widen_anything(client):
    """模型塞 owner_id / kb_id 也不会多看到东西 —— 范围只认服务端注入的那份。"""
    _H, uid, kb = _seed(client, "sql7")
    other_kb = "somebody-elses-kb"
    got = _sql(uid, kb).call({"table": "documents", "fields": ["kb_id"],
                              "filters": [{"field": "kb_id", "op": "eq", "value": other_kb}]})
    assert got["rows"] == []              # 过滤是**叠加**在属主范围之上的，不会绕过


def test_order_by_actually_sorts(client):
    """票面点名了「排序」—— 只测「非法字段被拒」不算数，得看真的排了。"""
    _H, uid, kb = _seed(client, "sql8", files=("b.txt", "a.txt", "c.txt"))
    tool = _sql(uid, kb)

    asc = [r["filename"] for r in tool.call(
        {"table": "documents", "fields": ["filename"],
         "order_by": {"field": "filename"}})["rows"]]
    desc = [r["filename"] for r in tool.call(
        {"table": "documents", "fields": ["filename"],
         "order_by": {"field": "filename", "desc": True}})["rows"]]

    assert asc == sorted(asc) and asc == ["a.txt", "b.txt", "c.txt"]
    assert desc == list(reversed(asc))


def test_group_by_defaults_to_desc_by_the_aggregate(client):
    """没点名排序时按聚合值倒序 —— 「哪个库文档最多」直接就能读。"""
    _H, uid, kb = _seed(client, "sql9", files=("a.txt", "b.txt"))
    rows = _sql(uid, kb).call({"table": "documents", "group_by": ["kb_id"],
                               "aggregate": {"fn": "count"}})["rows"]
    assert rows[0]["count"] == 2
