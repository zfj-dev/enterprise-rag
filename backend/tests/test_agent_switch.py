"""代理开关（票 15 / #22）：默认关、行为与今天一致；打开才走代理，且事件契约不变。"""
from __future__ import annotations

from app.services import chat_service
from tests.helpers import register_and_kb, sse_events, wait_until

DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。"

_AGENT_RESULT = {
    "answer": "代理给的答案：803.96 亿元。",
    "sources": [{"chunk_id": "c1", "doc_name": "年报.pdf", "page": 2, "text": "营收803.96亿元"}],
    "steps": [{"tool": "KbRetrieve", "arguments": {"query": "营收"}, "summary": "{}", "ms": 1.0,
               "ok": True}],
    "latency": {"total_ms": 12.0, "steps_ms": [1.0]},
    "stopped": "answered",
    "trace": {"self_check": "passed_with_citation", "citation_coverage": 1.0},
}


def _seeded(client, name: str):
    """注册 + 建库 + 传文档并等入库，返回 (H, kb_id, user, db, rt)。"""
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
    return H, kb, user, db, get_runtime()


def test_switch_is_off_by_default_and_the_agent_is_never_built(client, monkeypatch):
    """默认关：代理那套一行都不跑 —— 既有行为与既有测试都不受影响。"""
    called = []
    H, kb, user, db, rt = _seeded(client, "switch_off")
    try:
        monkeypatch.setattr(chat_service, "_run_agent_for",
                            lambda *a, **k: called.append(1))
        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
    finally:
        db.close()

    assert called == []
    assert "agent" not in out["trace"]


def test_switch_on_routes_the_answer_through_the_agent(client, monkeypatch):
    """打开开关：答案与来源都换成代理那份，自检结论进 trace。"""
    H, kb, user, db, rt = _seeded(client, "switch_on")
    try:
        monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
        monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: dict(_AGENT_RESULT))
        out = chat_service.answer(db, rt, user, kb, "营收多少？")
    finally:
        db.close()

    assert out["answer"] == _AGENT_RESULT["answer"]
    assert out["sources"] == _AGENT_RESULT["sources"]        # 来源是代理那份，不是确定性检索那份
    assert out["trace"]["agent"] is True
    assert out["trace"]["agent_stopped"] == "answered"
    assert out["trace"]["citation_coverage"] == 1.0
    assert out["trace"]["self_check"] == "passed_with_citation"


def test_the_sse_event_contract_is_unchanged_on_the_agent_path(client, monkeypatch):
    """前端不用改：仍是 sources -> delta -> done，字段一个不少。"""
    H, kb, user, db, rt = _seeded(client, "switch_sse")
    db.close()
    monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
    monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: dict(_AGENT_RESULT))

    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "营收多少？", "stream": True})
    events = sse_events(r.text)
    kinds = [e["type"] for e in events]

    assert kinds[0] == "sources" and kinds[-1] == "done" and "delta" in kinds
    assert events[0]["data"][0]["page"] == 2
    assert events[-1]["answer"] == _AGENT_RESULT["answer"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == _AGENT_RESULT["answer"]


def test_an_agent_failure_falls_back_instead_of_breaking_the_answer(client, monkeypatch):
    """代理那套自己炸了（工具集造不出来）不能把问答打成 500 —— 记日志、回退确定性链路。"""
    import app.mcp.registry as registry

    H, kb, user, db, rt = _seeded(client, "switch_boom")
    try:
        monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
        monkeypatch.setattr(registry, "build_registry",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("工具集炸了")))
        out = chat_service.answer(db, rt, user, kb, "文档里写了什么？")
    finally:
        db.close()

    assert out["answer"]                              # 仍然有答案
    assert "agent" not in out["trace"]                # 而且没有假装走了代理


class _NonFakeLLM:
    """非假模型：只为把「逐句校验引用」那条分支打开（真模型才走）。"""

    is_fake = False

    def stream(self, messages):
        yield ""


def test_the_agent_path_does_not_re_verify_the_coverage(client, monkeypatch):
    """代理自检（票 14）已经算过覆盖率 —— 别为同一件事再调一次模型。"""
    import app.core.citation as citation

    H, kb, user, db, rt = _seeded(client, "switch_no_reverify")
    calls = []
    try:
        monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
        monkeypatch.setattr(chat_service, "_run_agent_for", lambda *a, **k: dict(_AGENT_RESULT))
        monkeypatch.setattr(rt, "llm", _NonFakeLLM())
        monkeypatch.setattr(citation, "verify_claims",
                            lambda *a, **k: calls.append(1) or {"coverage": 0.0})
        out = chat_service.answer(db, rt, user, kb, "营收多少？")
    finally:
        db.close()

    assert calls == []                                   # 没有重复校验
    assert out["trace"]["citation_coverage"] == 1.0      # 沿用代理自检的结论


def test_tool_result_trimming_follows_the_compress_switch_and_the_exemption(client, monkeypatch):
    """工具结果清理（票 21）的开关：普通问题收、枚举/编号查询豁免（spec 0003 红线）、关压缩则全不收。"""
    import app.core.agent as agent_mod

    H, kb, user, db, rt = _seeded(client, "switch_trim")
    seen: dict = {}
    try:
        monkeypatch.setattr(chat_service.get_settings(), "agent_enabled", True)
        monkeypatch.setattr(agent_mod, "run_agent",
                            lambda q, **kw: seen.update(kw) or dict(_AGENT_RESULT))

        prep = chat_service.prepare(db, rt, user, kb, "这个文档讲了什么")
        assert prep.trace["compress_exempt"] is False
        chat_service._run_agent_for(db, rt, prep)
        assert seen["trim_tool_results"] is True                  # 普通问题：收

        prep = chat_service.prepare(db, rt, user, kb, "列出所有表格")
        assert prep.trace["compress_exempt"] is True
        chat_service._run_agent_for(db, rt, prep)
        assert seen["trim_tool_results"] is False                 # 枚举：豁免（列全不许漏项）

        monkeypatch.setattr(chat_service.get_settings(), "context_compress", False)
        prep = chat_service.prepare(db, rt, user, kb, "再问一次")
        chat_service._run_agent_for(db, rt, prep)
        assert seen["trim_tool_results"] is False                 # 关压缩：也不收
    finally:
        db.close()
