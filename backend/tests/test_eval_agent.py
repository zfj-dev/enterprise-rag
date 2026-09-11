"""评测适配器（票 15 / #22）：代理链路包成 answer_fn —— 评测核心零改动就能度量它。"""
from __future__ import annotations

from app.core.llm import ToolCall
from app.eval_core import run_eval
from tests.helpers import register_and_kb, wait_until

FACT = "803.96"
DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。"


class ScriptedLLM:
    """脚本化 LLM：先调一次 KbRetrieve 拿到来源，再据此作答 —— 无网络、不确定不来。"""

    is_fake = False

    def __init__(self, answer: str):
        self.answer = answer
        self.replies = [
            {"content": "", "tool_calls": [ToolCall(id="k1", name="KbRetrieve",
                                                    arguments={"query": "营收"})]},
            {"content": answer, "tool_calls": []},
        ]

    def stream(self, messages):
        yield self.answer          # verify_claims 走这条路（这里返回的不是 JSON，按降级处理）

    def chat_with_tools(self, messages, tools=None):
        if not tools:              # 收敛那一步：不给工具
            return {"content": self.answer, "tool_calls": []}
        if self.replies:
            return self.replies.pop(0)
        return {"content": self.answer, "tool_calls": []}


def _seeded(client, name: str):
    """注册 + 建库 + 传一份小文档并等入库，返回 (user, kb_id, db, rt)。"""
    from app.api.deps import get_runtime
    from app.db.session import SessionLocal
    from app.models.entities import User

    H, uid, kb = register_and_kb(client, name)
    up = client.post("/api/v1/documents?kb_id=%s" % kb, headers=H,
                     files={"file": ("annual.txt", DOC_TEXT, "text/plain")}).json()
    assert wait_until(lambda: client.get("/api/v1/documents/%s" % up["id"], headers=H)
                      .json().get("status") in ("indexed", "failed")), "文档未入库"

    db = SessionLocal()
    user = db.query(User).filter(User.id == uid).first()
    return user, kb, db, get_runtime()


def test_agent_answer_fn_is_shaped_for_the_eval_core(client):
    """适配器返回的就是核心要的 {answer, sources, citation_coverage} —— 评测代码一个字没改。"""
    from app.eval_agent import agent_answer_fn

    user, kb, db, rt = _seeded(client, "adapter_u")
    try:
        ask = agent_answer_fn(db, rt, user, kb, llm=ScriptedLLM("比亚迪2025年营业收入为803.96亿元。"))
        out = ask("比亚迪2025年营业收入是多少？")
    finally:
        db.close()

    assert out["answer"] == "比亚迪2025年营业收入为803.96亿元。"
    assert out["sources"] and all("chunk_id" in s for s in out["sources"])
    assert out["sources"][0]["page"] == 1            # 来源带页码，核心的 page_hit 才有分母
    assert "citation_coverage" in out


def test_the_eval_core_measures_the_agent_link_unchanged(client):
    """同一条黄金集交给核心跑代理链路，事实命中 / 页码都算得出来。"""
    from app.eval_agent import agent_answer_fn

    user, kb, db, rt = _seeded(client, "adapter_core")
    try:
        ask = agent_answer_fn(db, rt, user, kb, llm=ScriptedLLM("比亚迪2025年营业收入为803.96亿元。"))
        rep = run_eval([{"question": "比亚迪2025年营业收入是多少？", "expect": FACT, "page": 1}], ask)
    finally:
        db.close()

    assert rep.fact_rate == 1.0
    assert rep.page_rate == 1.0
    assert rep.positives[0].grounded is True          # 期望事实在随答案返回的来源里


def test_deterministic_answer_fn_is_shaped_for_the_eval_core(client):
    """确定性链路也走同一个缝 —— 两条链路同形，才谈得上对比。"""
    from app.eval_agent import deterministic_answer_fn

    user, kb, db, rt = _seeded(client, "adapter_det")
    try:
        out = deterministic_answer_fn(db, rt, user, kb)("文档里写了什么？")
    finally:
        db.close()

    assert set(out) == {"answer", "sources", "citation_coverage", "context"}
    assert out["answer"]
    assert out["sources"] and out["sources"][0]["page"] == 1


def test_agent_answer_fn_takes_an_injected_transport(client):
    """工具集可注入 —— 单测不必起真实 MCP 进程，也不必碰真实索引。"""
    from app.core.tools import Tool, ToolRegistry
    from app.eval_agent import agent_answer_fn
    from app.mcp.client import InProcessTransport

    tool = Tool(name="KbRetrieve", description="检索",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: {"sources": [{"chunk_id": "c1", "doc_name": "x.pdf",
                                                   "page": 2, "text": "甲", "score": 1.0}]})
    user, kb, db, rt = _seeded(client, "adapter_inj")
    try:
        ask = agent_answer_fn(db, rt, user, kb, llm=ScriptedLLM("甲。"),
                              transport=InProcessTransport(ToolRegistry(tools=[tool])))
        out = ask("问")
    finally:
        db.close()

    assert out["answer"] == "甲。"
    assert [(s["doc_name"], s["page"]) for s in out["sources"]] == [("x.pdf", 2)]


def test_deterministic_answer_fn_ignores_the_global_switch(client, monkeypatch):
    """基准链路不能被全局代理开关劫持 —— 否则两列都是代理，对比就是假的（审查抓到）。"""
    from app.config import get_settings
    from app.eval_agent import deterministic_answer_fn
    from app.services import chat_service

    called = []
    user, kb, db, rt = _seeded(client, "adapter_noagent")
    try:
        monkeypatch.setattr(get_settings(), "agent_enabled", True)
        monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: called.append(1))
        out = deterministic_answer_fn(db, rt, user, kb)("文档里写了什么？")
    finally:
        db.close()

    assert called == []          # 开关开着，基准链路照样不走代理
    assert out["answer"]
