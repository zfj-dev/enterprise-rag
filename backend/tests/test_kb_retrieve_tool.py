"""KbRetrieve 工具（票 12 / #19）：范围服务端注入、检索与问答同质、越权取不到。"""
from __future__ import annotations

import pytest

from app.core.tools import ToolError
from app.mcp.client import InProcessTransport
from tests.helpers import register_and_kb, wait_until

SAME_TEXT = "比亚迪安全手册，关于电池和充电的规范说明。"


def _seed(client, name, text=SAME_TEXT):
    """注册 + 建库 + 传一份小文档并等入库，返回 (headers, uid, kb_id)。"""
    H, uid, kb = register_and_kb(client, name)
    up = client.post("/api/v1/documents?kb_id=%s" % kb, headers=H,
                     files={"file": ("manual.txt", text, "text/plain")}).json()
    assert wait_until(lambda: client.get("/api/v1/documents/%s" % up["id"],
                                         headers=H).json().get("status") in ("indexed", "failed")), \
        "文档未入库"
    return H, uid, kb


def _transport(uid, kb):
    """用**应用自己那个**运行时（文档就索引在它里面）造工具集。

    测试经由**传输层**调用（与代理走的是同一条缝），而不是直接捅 handler。
    """
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.mcp.registry import build_registry
    from app.models.entities import User

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uid).first()
        return InProcessTransport(build_registry(db, get_runtime(), user, kb))
    finally:
        db.close()


def _own_chunk_ids(kb: str, owner: str) -> set:
    """某个 (库, 属主) 名下的全部块 id —— 用来正向证明「拿回来的确实都是自己的」。"""
    from app.db.session import SessionLocal
    from app.models.entities import Chunk

    db = SessionLocal()
    try:
        return {c.id for c in db.query(Chunk)
                .filter(Chunk.kb_id == kb, Chunk.owner_id == owner).all()}
    finally:
        db.close()


def test_returns_candidates_with_stable_ids_and_real_pages(client):
    _H, uid, kb = _seed(client, "kbret1")
    got = _transport(uid, kb).call_tool("KbRetrieve", {"query": "电池充电规范"})

    assert got["sources"], "应该检索到内容"
    first = got["sources"][0]
    assert set(first) == {"chunk_id", "doc_id", "doc_name", "page", "text", "score"}
    assert first["chunk_id"]                           # 稳定 chunk_id
    assert first["doc_name"] == "manual.txt"
    assert first["page"] >= 1                          # 真实页码
    assert isinstance(first["score"], float)
    assert first["text"] and "[文档概要]" not in first["text"]   # 与问答 sources 同质：前缀已剥


def test_range_parameters_from_the_model_are_ignored(client):
    """模型就算塞 kb_id / owner_id，也一律不看 —— 范围只认闭包带进来的那份。"""
    _H, uid, kb = _seed(client, "kbret2")
    got = _transport(uid, kb).call_tool("KbRetrieve", {"query": "电池", "kb_id": "别人的库",
                                                       "owner_id": "攻击者", "top_k": 3})

    assert got["sources"]
    assert all(s["doc_name"] == "manual.txt" for s in got["sources"])


def test_top_k_is_clamped_and_invalid_falls_back(client):
    _H, uid, kb = _seed(client, "kbret3")
    transport = _transport(uid, kb)

    assert len(transport.call_tool("KbRetrieve", {"query": "电池", "top_k": 1})["sources"]) <= 1
    assert len(transport.call_tool("KbRetrieve", {"query": "电池", "top_k": "乱写的"})["sources"]) <= 5
    assert len(transport.call_tool("KbRetrieve", {"query": "电池", "top_k": 9999})["sources"]) <= 20


def test_missing_query_is_a_tool_error(client):
    _H, uid, kb = _seed(client, "kbret4")
    with pytest.raises(ToolError):
        _transport(uid, kb).call_tool("KbRetrieve", {"query": "   "})


def test_every_returned_chunk_belongs_to_the_callers_own_scope(client):
    """越权取不到 —— 两条同内容的文档，只有属主/库不同，拿回来的必须全是自己的。

    只断言「两边不重叠」证不了什么（doc_id 本来就不同）；这里正向断言
    「每个块都在我自己的 (库, 属主) 名下」，owner 过滤一旦失效就会挂。
    """
    _H1, uid1, kb1 = _seed(client, "kbretA")
    _H2, uid2, kb2 = _seed(client, "kbretB")            # **同一份文本**，只有属主/库不同

    got = _transport(uid1, kb1).call_tool("KbRetrieve", {"query": "电池充电规范"})["sources"]
    assert got
    assert {s["chunk_id"] for s in got} <= _own_chunk_ids(kb1, uid1)
    assert not ({s["chunk_id"] for s in got} & _own_chunk_ids(kb2, uid2))


def test_enumeration_intent_is_reused_by_the_tool(client):
    """票 12 第 3 条：枚举意图 / 具体编号处理与普通问答**共用同一份**，工具也走得到。"""
    table = "表 3.1 实验环境配置\n\n| 项目 | 值 |\n|---|---|\n| GPU | RTX 3060 |\n"
    _H, uid, kb = _seed(client, "kbret5", text=table)

    got = _transport(uid, kb).call_tool("KbRetrieve", {"query": "列出所有表格内容"})

    assert got["sources"]
    assert any("|" in s["text"] for s in got["sources"])       # 表格块被枚举分支前置进来了
