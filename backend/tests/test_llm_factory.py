"""BYOK 按请求解析（票 31）：LLM 从全局单例改为按用户解析，未配置回落全局。

缝：Runtime 上注入的 llm_factory + user_llm_config_store。
stub 工厂 + 内存配置库 → 确定性、无网络。
"""
from __future__ import annotations

import time

from tests.helpers import sse_events

from app.core.byok import InMemoryUserLLMConfigStore, LLMConfig
from app.core.container import build_runtime
from app.core.llm import LLM

CFG = LLMConfig(base_url="https://api.example.com/v1", api_key="sk-x", model="my-model")


class StubLLM(LLM):
    """记号化的假模型：产出可辨识的文本，便于断言"到底用了哪一个"。"""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list = []

    def stream(self, messages):
        self.calls.append(messages)
        yield self.tag


class RecordingFactory:
    def __init__(self, tag: str = "这是自带模型产出的答案。"):
        self.tag = tag
        self.built: list[LLMConfig] = []

    def build(self, cfg: LLMConfig) -> LLM:
        self.built.append(cfg)
        return StubLLM(self.tag)


def _runtime(factory=None, store=None):
    rt = build_runtime()
    if factory is not None:
        rt.llm_factory = factory
    if store is not None:
        rt.user_llm_config_store = store
    return rt


def _deltas(resp_text: str) -> str:
    return "".join(e.get("text", "") for e in sse_events(resp_text) if e.get("type") == "delta")


# ---------- 解析 ----------

def test_unconfigured_user_falls_back_to_global():
    """未配置 BYOK → 用服务端全局 LLM，且工厂根本不被调用。"""
    f = RecordingFactory()
    rt = _runtime(f, InMemoryUserLLMConfigStore())
    assert rt.llm_for("u1") is rt.llm
    assert f.built == []


def test_configured_user_uses_own_llm():
    store = InMemoryUserLLMConfigStore()
    store.set("u1", CFG)
    f = RecordingFactory()
    rt = _runtime(f, store)
    llm = rt.llm_for("u1")
    assert isinstance(llm, StubLLM)
    assert f.built == [CFG]


def test_user_isolation():
    """A 配了自带模型，不影响 B —— B 仍回落全局。"""
    store = InMemoryUserLLMConfigStore()
    store.set("u1", CFG)
    rt = _runtime(RecordingFactory(), store)
    assert isinstance(rt.llm_for("u1"), StubLLM)
    assert rt.llm_for("u2") is rt.llm


def test_delete_reverts_to_global():
    store = InMemoryUserLLMConfigStore()
    store.set("u1", CFG)
    store.delete("u1")
    rt = _runtime(RecordingFactory(), store)
    assert rt.llm_for("u1") is rt.llm


# ---------- 请求链路真的用了它（集成） ----------

def _seed_user(client, name: str, kb_name: str):
    """注册用户 + 建库 + 上传一份小文档并等入库（有引用源，答案才不会被 no-source 兜底替换）。"""
    from app.db.session import SessionLocal
    from app.models.entities import User

    tok = client.post("/api/v1/auth/register",
                      json={"username": name, "password": "pw123456"}).json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}
    kb = client.post("/api/v1/knowledge", json={"name": kb_name, "description": ""}, headers=H).json()["id"]
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                     files={"file": ("manual.txt", "比亚迪安全手册，关于电池和充电的规范说明。", "text/plain")}).json()
    for _ in range(60):
        d = client.get(f"/api/v1/documents/{up['id']}", headers=H).json()
        if d.get("status") in ("indexed", "failed"):
            assert d["status"] == "indexed", d
            break
        time.sleep(0.1)
    else:
        raise AssertionError("文档未入库")

    db = SessionLocal()
    try:
        uid = db.query(User).filter(User.username == name).first().id
    finally:
        db.close()
    return H, kb, uid


def test_chat_stream_uses_user_llm(client):
    """带自带配置的用户，其回答由该用户的 LLM 产出；另一个用户仍走全局。"""
    import app.api.deps as deps

    H, kb, uid = _seed_user(client, "byokz", "kb")
    H2, kb2, _ = _seed_user(client, "nobyk", "kb2")

    store = InMemoryUserLLMConfigStore()
    store.set(uid, CFG)
    deps._runtime = _runtime(RecordingFactory("这是自带模型产出的答案。"), store)

    ans = _deltas(client.post("/api/v1/chat/stream", headers=H,
                              json={"kb_id": kb, "question": "电池 规范", "stream": True}).text)
    assert "自带模型" in ans

    ans2 = _deltas(client.post("/api/v1/chat/stream", headers=H2,
                               json={"kb_id": kb2, "question": "电池 规范", "stream": True}).text)
    assert "自带模型" not in ans2


def test_user_llm_bypasses_semantic_cache(client):
    """自带模型的用户不吃共享语义缓存 —— 否则会拿到全局模型生成的答案。"""
    import app.api.deps as deps

    H, kb, uid = _seed_user(client, "byokc", "kbc")
    store = InMemoryUserLLMConfigStore()
    store.set(uid, CFG)
    deps._runtime = _runtime(RecordingFactory(), store)

    def ask():
        r = client.post("/api/v1/chat/stream", headers=H,
                        json={"kb_id": kb, "question": "电池 规范", "stream": True})
        hits = [e.get("cache_hit") for e in sse_events(r.text) if e.get("type") == "done"]
        return r.status_code, (hits[-1] if hits else None)

    s1, hit1 = ask()
    s2, hit2 = ask()   # 完全相同的问题：全局模型会命中缓存，自带模型必须不命中
    assert s1 == 200 and s2 == 200
    assert hit1 is False
    assert hit2 is False
