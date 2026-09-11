"""BYOK 凭据存储（票 32 / #39）：key 加密落库、从不回显、可轮换可删除。

明文 key 只在「填进来」与「真要调模型」两处出现 —— 库、响应体、日志里都不该有它。
"""
from __future__ import annotations

import logging

import pytest

from app.core.byok import DbUserLLMConfigStore, InMemoryUserLLMConfigStore, LLMConfig, key_tail
from tests.helpers import register_and_kb

SECRET = "server-side-passphrase-用于派生加密密钥"
KEY = "sk-abcdefghijklmnop-1234"          # 尾号 1234
PUBLIC = "93.184.216.34"                  # 公网 IP：SSRF 校验（票 33）要解析域名，测试里注入它


def _use_encrypted_store(rt):
    """装上加密存储，并让 SSRF 的域名解析离线可控 —— 否则 PUT 会去真的查 DNS。"""
    rt.user_llm_config_store = DbUserLLMConfigStore(SECRET)
    rt.url_resolver = lambda host: [PUBLIC]
    return rt


def _cfg(model="qwen-plus", key=KEY):
    return LLMConfig(base_url="https://api.example.com/v1", api_key=key, model=model)


# ---------- 加密落库 ----------

def test_the_plaintext_key_never_lands_in_the_database(client):
    """库里只该有密文与尾号 —— 数据库文件被看到时，凭据不能直接可用。"""
    from app.db.session import SessionLocal
    from app.models.entities import UserLLMConfig

    store = DbUserLLMConfigStore(SECRET)
    store.set("u1", _cfg())

    db = SessionLocal()
    try:
        row = db.get(UserLLMConfig, "u1")
        assert row.key_cipher and KEY not in row.key_cipher     # 落库的是密文
        assert row.key_tail == "1234"                            # 只留尾号
    finally:
        db.close()

    assert store.get("u1").api_key == KEY                        # 取出来能还原


def test_a_weak_or_absent_encryption_key_is_refused_at_construction():
    """加密口令没有默认值、也没有弱默认 —— 空口令与「一字符口令」都当场拒绝，绝不退化成明文存储。"""
    for weak in ("", "1", "short"):
        with pytest.raises(ValueError):
            DbUserLLMConfigStore(weak)


def test_the_same_key_seals_differently_each_time():
    """每行自己的随机盐：同一个 key 两次落库得到不同密文（也挡掉了「同口令同密钥」的比对）。"""
    from app.db.session import SessionLocal
    from app.models.entities import UserLLMConfig

    store = DbUserLLMConfigStore(SECRET)
    store.set("u1", _cfg())
    store.set("u2", _cfg())
    db = SessionLocal()
    try:
        c1 = db.get(UserLLMConfig, "u1").key_cipher
        c2 = db.get(UserLLMConfig, "u2").key_cipher
    finally:
        db.close()

    assert c1 != c2 and "$" in c1                      # 盐与密文分开存
    assert store.get("u1").api_key == store.get("u2").api_key == KEY


def test_a_changed_passphrase_is_treated_as_not_configured(client):
    """换了口令：旧密文解不开 —— 按「没配」处理（回落全局），而不是把问答打崩。"""
    DbUserLLMConfigStore(SECRET).set("u1", _cfg())

    assert DbUserLLMConfigStore("另一个足够长的口令-alternate").get("u1") is None


def test_stores_are_per_user(client):
    store = DbUserLLMConfigStore(SECRET)
    store.set("u1", _cfg(model="m1"))
    store.set("u2", _cfg(model="m2"))

    assert store.get("u1").model == "m1"
    assert store.get("u2").model == "m2"
    store.delete("u1")
    assert store.get("u1") is None and store.get("u2") is not None


# ---------- 不回显 ----------

def test_the_public_view_has_no_room_for_the_key(client):
    store = DbUserLLMConfigStore(SECRET)
    store.set("u1", _cfg())

    view = store.public_view("u1")

    assert set(view) == {"base_url", "model", "key_tail", "updated_at"}
    assert view["key_tail"] == "1234" and KEY not in str(view)


def test_repr_never_contains_the_key():
    """日志里顺手打一个 config 就能泄漏 —— 所以 key 字段压根不进 repr（连字段名都不出现）。"""
    text = repr(_cfg())

    assert KEY not in text
    assert "1234" not in text                 # 尾号也不给：它是**回显接口**专用的
    assert "qwen-plus" in text and "api.example.com" in text   # 非敏感部分照常可读


def test_logging_a_config_does_not_leak_the_key(caplog):
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("byok.test").info("当前配置：%s", _cfg())

    assert KEY not in caplog.text


def test_the_api_response_body_contains_no_plaintext_key(client):
    """对**响应体本身**断言：不是「字段里没有」，而是整段文本里找不到那个 key。"""
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_api")
    _use_encrypted_store(get_runtime())

    put = client.put("/api/v1/llm/config", headers=H,
                     json={"base_url": "https://api.example.com/v1", "key": KEY, "model": "qwen-plus"})
    got = client.get("/api/v1/llm/config", headers=H)

    assert put.status_code == 200 and got.status_code == 200
    assert KEY not in put.text and KEY not in got.text
    assert got.json()["key_tail"] == "1234"
    assert got.json()["configured"] is True


def test_a_validation_error_does_not_echo_the_body_back(client):
    """**校验失败的响应也不许含明文 key**：FastAPI 默认会把整个 body 塞进 422 详情里。"""
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_422")
    _use_encrypted_store(get_runtime())

    # 故意漏掉 model —— 触发 422
    r = client.put("/api/v1/llm/config", headers=H,
                   json={"base_url": "https://api.example.com/v1", "key": KEY})

    assert r.status_code == 422
    assert KEY not in r.text                      # 对**响应体**直接断言，不是只看某个字段
    assert "model" in r.text                       # 原因照常说清楚


def test_the_view_says_whether_it_survives_a_restart(client):
    """只存内存时如实说「重启就没了」，别让人以为已经存住了。"""
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_persist")
    rt = get_runtime()

    rt.user_llm_config_store = InMemoryUserLLMConfigStore()
    rt.url_resolver = lambda host: [PUBLIC]
    assert client.get("/api/v1/llm/config", headers=H).json()["persistent"] is False

    _use_encrypted_store(rt)
    assert client.get("/api/v1/llm/config", headers=H).json()["persistent"] is True


def test_an_unconfigured_user_gets_an_empty_view(client):
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_empty")
    _use_encrypted_store(get_runtime())

    body = client.get("/api/v1/llm/config", headers=H).json()

    assert body["configured"] is False and body["key_tail"] == "" and body["base_url"] == ""


# ---------- 生效、轮换、删除 ----------

class StubFactory:
    """确定性工厂：不构造真模型，只记下拿到的配置。"""

    def __init__(self):
        self.built = []

    def build(self, cfg):
        self.built.append(cfg)
        return "LLM(%s)" % cfg.model


def test_a_configured_user_uses_their_own_model_and_rotation_takes_effect(client):
    """轮换就是覆盖：新凭据生效，旧的那份随即失效。"""
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_rotate")
    rt = get_runtime()
    _use_encrypted_store(rt)
    rt.llm_factory = StubFactory()

    client.put("/api/v1/llm/config", headers=H,
               json={"base_url": "https://a/v1", "key": "sk-old-0001", "model": "m-old"})
    assert rt.llm_for(uid) == "LLM(m-old)"

    client.put("/api/v1/llm/config", headers=H,
               json={"base_url": "https://b/v1", "key": "sk-new-0002", "model": "m-new"})
    assert rt.llm_for(uid) == "LLM(m-new)"                       # 旧凭据立刻不再生效
    assert rt.user_llm_config_store.get(uid).api_key == "sk-new-0002"


def test_deleting_falls_back_to_the_global_model(client):
    from app.api.deps import get_runtime

    H, uid, kb = register_and_kb(client, "byok_delete")
    rt = get_runtime()
    _use_encrypted_store(rt)
    rt.llm_factory = StubFactory()

    client.put("/api/v1/llm/config", headers=H,
               json={"base_url": "https://a/v1", "key": "sk-x-0003", "model": "m-own"})
    assert rt.llm_for(uid) != rt.llm

    client.delete("/api/v1/llm/config", headers=H)

    assert rt.llm_for(uid) is rt.llm                             # 立刻回落服务端全局
    assert client.get("/api/v1/llm/config", headers=H).json()["configured"] is False


def test_one_users_config_does_not_affect_another(client):
    from app.api.deps import get_runtime

    H_a, uid_a, _ = register_and_kb(client, "byok_iso_a")
    H_b, uid_b, _ = register_and_kb(client, "byok_iso_b")
    rt = get_runtime()
    _use_encrypted_store(rt)
    rt.llm_factory = StubFactory()

    client.put("/api/v1/llm/config", headers=H_a,
               json={"base_url": "https://a/v1", "key": "sk-a-0001", "model": "m-a"})

    assert rt.llm_for(uid_a) == "LLM(m-a)"
    assert rt.llm_for(uid_b) is rt.llm                           # 乙没配 → 走全局
    assert client.get("/api/v1/llm/config", headers=H_b).json()["configured"] is False


# ---------- 默认实现 ----------

def test_the_default_store_is_in_memory_so_tests_stay_isolated():
    """默认内存实现：谁都没配过 → 全部回落全局，行为与今天一致（也不需要加密口令）。"""
    from app.core.container import Runtime  # noqa: F401 —— 只为说明默认值

    store = InMemoryUserLLMConfigStore()
    store.set("u1", _cfg())

    assert store.public_view("u1")["key_tail"] == key_tail(KEY)
    store.delete("u1")
    assert store.get("u1") is None
