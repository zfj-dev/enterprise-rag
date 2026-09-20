"""上线前安全审查（2026-09-20）修复的回归用例。

每条都对着「改坏就会红」的那个行为写，不是补覆盖率：

- B2 公开密钥：仓库里出现过的 SECRET_KEY 必须**拒绝启动**（它签的是 JWT）；
- B1 管理员口令：`ADMIN_PASSWORD` 生效、真实模式不接受短口令、也不回落公开口令；
- B1 改密：改完旧口令失效、新口令可用；未登录不给改；
- H1 上传：超限在**读的过程中**就抛，而不是把整个流拖进内存再判；
- H2 client-error：按 IP 限流、写日志前剥掉换行（防日志注入）。
"""
from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException

from app.config import Settings

# ---------- B2：公开 SECRET_KEY 一律拒绝 ----------


@pytest.mark.parametrize("bad", [
    "dev-secret-change-me-0123456789abcdef",              # config.py 的代码默认值
    "change-me",                                          # .env.example
    "change-me-in-prod",                                  # deploy/docker-compose.yml
    "change-me-strong-0123456789abcdef0123456789",        # deploy/.env.example
    "dev-rag-secret-0123456789abcdef0123456789",          # scripts/run_real.ps1.example
])
def test_secret_key_public_values_are_rejected(bad):
    """这些值在仓库里是公开的 —— 用它们签 JWT 等于把管理员身份发给所有人。

    原来只在 USE_REAL=true 时拦，而 deploy/docker-compose.yml 恰好是 USE_REAL=false，
    于是那份「开箱即用」的编排带着公开密钥就起来了。
    """
    with pytest.raises(Exception):
        Settings(_env_file=None, secret_key=bad, use_real=False)


def test_secret_key_too_short_is_rejected():
    """黑名单挡不住随手写的密钥，长度闸兜底。"""
    with pytest.raises(Exception):
        Settings(_env_file=None, secret_key="x" * 31)


def test_secret_key_strong_is_accepted():
    s = Settings(_env_file=None, secret_key="x" * 32)
    assert len(s.secret_key) == 32


# ---------- B1：管理员口令 ----------


def test_admin_password_is_read_from_dotenv_too(tmp_path):
    """`ADMIN_PASSWORD` 必须能从 `.env` 读进 Settings，而不只是从环境变量。

    修复的第一版是裸读 `os.environ`，而 pydantic-settings 把 `.env` 灌进 Settings、
    **不**写进 `os.environ` —— 结果按 `.env.example` 配好的人被**静默**回落成公开口令。
    这条用例把「必须走配置字段」钉住（与 core/tokenizer.py 的 HF_ENDPOINT 同类）。
    """
    env = tmp_path / ".env"
    env.write_text("ADMIN_PASSWORD=from-dotenv-123456\n", encoding="utf-8")

    s = Settings(_env_file=str(env), secret_key="x" * 40)
    assert s.admin_password == "from-dotenv-123456"


def test_seed_admin_uses_configured_password(monkeypatch):
    """配了 ADMIN_PASSWORD 就用它 —— 不再无条件写死公开口令。"""
    import app.main as m
    from app.db.session import SessionLocal
    from app.models.entities import User
    from app.utils.security import verify_password

    class _S:
        use_real = False
        admin_password = "env-set-admin-pw-123"

    monkeypatch.setattr(m, "get_settings", lambda: _S())

    m._seed_admin()

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "admin").first()
        assert u is not None and u.role == "admin"
        assert verify_password("env-set-admin-pw-123", u.password_hash)
        assert not verify_password("admin123", u.password_hash)
    finally:
        db.close()


def test_seed_admin_real_mode_rejects_short_password(monkeypatch):
    """真实模式配了个短口令 → 直接拒绝启动，别让弱口令上线。"""
    import app.main as m

    class _S:
        use_real = True
        admin_password = "short"

    monkeypatch.setattr(m, "get_settings", lambda: _S())

    with pytest.raises(RuntimeError):
        m._seed_admin()


def test_seed_admin_real_mode_refuses_to_create_without_password(monkeypatch):
    """真实模式没配 ADMIN_PASSWORD、又确实要建号 → **拒绝启动**，绝不回落公开口令。

    安全审查 F11：不再「随机生成后打印到日志」—— 容器日志常被采集，把初始口令写进去
    等于换个地方泄漏；真实模式本来也没有「合理的默认口令」可言。
    """
    import app.main as m
    from app.db.session import SessionLocal
    from app.models.entities import User

    class _S:
        use_real = True
        admin_password = ""

    monkeypatch.setattr(m, "get_settings", lambda: _S())

    with pytest.raises(RuntimeError):
        m._seed_admin()

    db = SessionLocal()
    try:
        assert db.query(User).filter(User.username == "admin").first() is None, "一个都不该建"
    finally:
        db.close()


def test_seed_admin_real_mode_keeps_booting_when_admin_already_exists(monkeypatch):
    """库里已有 admin 时**不报错** —— 否则这条规则会让既有部署起不来。

    （口令轮换走 /auth/change-password，不靠启动时重建。）
    """
    import app.main as m
    from app.db.session import SessionLocal
    from app.models.entities import User

    # 先按演示模式建出一个 admin（等价于「上一版部署留下的那个」）
    class _Demo:
        use_real = False
        admin_password = ""

    monkeypatch.setattr(m, "get_settings", lambda: _Demo())
    m._seed_admin()

    # 再以真实模式启动：应该什么都不做、也不抛
    class _Real:
        use_real = True
        admin_password = ""

    monkeypatch.setattr(m, "get_settings", lambda: _Real())
    m._seed_admin()

    db = SessionLocal()
    try:
        assert db.query(User).filter(User.username == "admin").count() == 1
    finally:
        db.close()


# ---------- B1：改密 ----------


def test_change_password_rotates_credential(client, auth_headers):
    h = auth_headers("pw_user", password="pw123456")

    # 旧口令填错 → 400，且**原来的口令仍然有效**（没被改坏）
    bad = client.post("/api/v1/auth/change-password", headers=h,
                      json={"old_password": "wrong", "new_password": "brand-new-pass-123"})
    assert bad.status_code == 400
    assert client.post("/api/v1/auth/login",
                       json={"username": "pw_user", "password": "pw123456"}).status_code == 200

    ok = client.post("/api/v1/auth/change-password", headers=h,
                     json={"old_password": "pw123456", "new_password": "brand-new-pass-123"})
    assert ok.status_code == 200

    assert client.post("/api/v1/auth/login",
                       json={"username": "pw_user", "password": "pw123456"}).status_code == 401
    assert client.post("/api/v1/auth/login",
                       json={"username": "pw_user", "password": "brand-new-pass-123"}).status_code == 200


def test_change_password_requires_login(client):
    """拿不到令牌就改不了别人的口令。"""
    r = client.post("/api/v1/auth/change-password",
                    json={"old_password": "whatever", "new_password": "brand-new-pass-123"})
    assert r.status_code == 401


def test_change_password_rejects_weak_new_password(client, auth_headers):
    """新口令与注册共用同一个下限（`MIN_PASSWORD_CHARS`），短于它一律 422。"""
    h = auth_headers("pw_weak")
    r = client.post("/api/v1/auth/change-password", headers=h,
                    json={"old_password": "pw123456", "new_password": "short"})
    assert r.status_code == 422


# ---------- H1：上传边读边限长 ----------


class _FakeStream:
    """一个「无限长」的字节流；记录被真正读走了多少。"""

    def __init__(self, total: int):
        self.total = total
        self.read_bytes = 0

    def read(self, n: int | None = None) -> bytes:
        take = self.total - self.read_bytes if n is None else min(n, self.total - self.read_bytes)
        if take <= 0:
            return b""
        self.read_bytes += take
        return b"A" * take


class _FakeUpload:
    def __init__(self, total: int):
        self.file = _FakeStream(total)


def test_read_capped_stops_while_reading_not_after():
    """超限必须在**读的过程中**就抛。

    原来的 `file.file.read()` 是把整个文件拖进内存**之后**才判大小 —— 校验发生在内存
    已经吃满之后，一个超大文件就能把进程打爆（安全审查 H1）。
    """
    from app.api.v1 import documents as docs

    up = _FakeUpload(total=64 * 1024 * 1024)          # 流有 64MB
    with pytest.raises(HTTPException) as ei:
        docs._read_capped(up, max_bytes=1 << 20)      # 上限 1MB
    assert ei.value.status_code == 413
    # 关键断言：只读走「上限 + 一个块」就停了，没把 64MB 全拖进来
    assert up.file.read_bytes <= (1 << 20) + docs._READ_CHUNK


def test_read_capped_passes_through_when_under_limit():
    from app.api.v1 import documents as docs

    up = _FakeUpload(total=100)
    assert docs._read_capped(up, max_bytes=1 << 20) == b"A" * 100


def test_read_capped_unlimited_when_no_cap():
    """max_upload_mb=0 表示不限制（沿用原语义）。"""
    from app.api.v1 import documents as docs

    up = _FakeUpload(total=5000)
    assert len(docs._read_capped(up, max_bytes=0)) == 5000


# ---------- H2：client-error 限流 + 防日志注入 ----------


def test_client_error_is_rate_limited(client):
    """未鉴权端点按 IP 限流 —— 否则它是一条把磁盘写满的路径。"""
    from app import main as m

    m._client_error_limiter.clear()
    codes = [client.post("/api/v1/client-error", json={"msg": f"m{i}"}).json().get("ok")
             for i in range(m._CE_MAX + 1)]

    assert all(c is True for c in codes[: m._CE_MAX])
    assert codes[m._CE_MAX] is False
    # 超限走 429，与本文件其它错误一致 —— 而不是 200 配一个 ok:false
    assert client.post("/api/v1/client-error", json={"msg": "x"}).status_code == 429


def test_client_error_strips_newlines_before_logging(client):
    """换行必须剥掉，否则上报内容能伪造出整行日志（日志注入）。

    `_error_logger.propagate = False`，所以不能靠 caplog —— 直接挂一个 handler 收。
    """
    from app import main as m

    m._client_error_limiter.clear()
    seen: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    h = _H()
    m._error_logger.addHandler(h)
    try:
        r = client.post("/api/v1/client-error",
                        json={"msg": "boom\r\n2099-01-01 ERROR 伪造的一行"})
    finally:
        m._error_logger.removeHandler(h)

    assert r.status_code == 200
    assert seen, "应该记了一条"
    assert "\n" not in seen[0] and "\r" not in seen[0]


# ---------- F1：请求体大小上限 ----------


def test_oversized_body_is_rejected_before_it_is_read(client):
    """超限的请求体在读进内存**之前**就被 413 挡掉。

    字段级的 `max_length` 不省内存（Pydantic 是在 body 读完**之后**才校验的），
    真正兜底的是 `BodySizeLimitMiddleware`（安全审查 F1）。
    """
    from app.config import get_settings

    limit = get_settings().max_body_mb * 1024 * 1024
    r = client.post("/api/v1/client-error", json={"msg": "A" * (limit + 1024)})
    assert r.status_code == 413


def test_body_limit_rejection_carries_cors_headers(client):
    """413 要带上跨域响应头 —— 中间件排在 CORS **内层**才做得到。

    排在最外层的话，跨域前端看到的是一次「CORS 失败」而不是 413，排查会被带偏。
    """
    from app.config import get_settings

    limit = get_settings().max_body_mb * 1024 * 1024
    r = client.post("/api/v1/client-error", json={"msg": "A" * (limit + 1024)},
                    headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == "*"


def test_upload_path_gets_a_larger_body_budget_and_is_not_buffered():
    """上传路径按前缀放宽，而且**大上限的路径不走内存缓冲**。

    缓冲会把几十 MB 从「Starlette spool 到磁盘」变成「占着内存」，等于自己造一个新的
    OOM 面；大请求交给 Content-Length 预判 + 业务层的边读边限长（`_read_capped`）。
    """
    from app.config import get_settings
    from app.main import app
    from app.utils import bodylimit
    from app.utils.bodylimit import BodySizeLimitMiddleware

    mw = [m for m in app.user_middleware if m.cls is BodySizeLimitMiddleware][0]
    upload_limit = mw.kwargs["overrides"][get_settings().api_prefix + "/documents"]
    assert upload_limit > get_settings().max_body_mb * 1024 * 1024
    assert upload_limit > bodylimit._BUFFER_MAX, "上传上限应当落在「不缓冲」那一档"


def test_default_body_limit_stays_within_the_buffered_range():
    """通用上限必须 ≤ `_BUFFER_MAX`，否则 JSON 接口那一道会**静默失效**（走了不缓冲分支）。

    这是个防手滑的闸：谁把 MAX_BODY_MB 调到 2MB 以上，这里就会红。
    """
    from app.config import get_settings
    from app.utils import bodylimit

    assert get_settings().max_body_mb * 1024 * 1024 <= bodylimit._BUFFER_MAX


def test_client_error_rejects_oversized_field(client):
    """字段级上限仍在：msg 超过 2000 字符 → 422（body 本身没超中间件的限）。"""
    assert client.post("/api/v1/client-error", json={"msg": "x" * 5000}).status_code == 422


# ---------- F2：删知识库要清磁盘 ----------


def test_delete_kb_removes_files_from_disk(client):
    """删知识库要连磁盘上的原文件一起删 —— 否则「已删除」的文档还躺在盘上（安全审查 F2）。"""
    import os

    from app.db.session import SessionLocal
    from app.models.entities import Document
    from tests.helpers import register_and_kb, wait_until

    H, _, kb = register_and_kb(client, "kbdel")
    up = client.post(f"/api/v1/documents?kb_id={kb}", headers=H,
                     files={"file": ("a.txt", "比亚迪安全手册内容", "text/plain")}).json()
    assert wait_until(lambda: client.get(f"/api/v1/documents/{up['id']}", headers=H)
                      .json().get("status") in ("indexed", "failed")), "文档未入库"

    db = SessionLocal()
    try:
        path = db.query(Document).filter(Document.id == up["id"]).first().file_path
    finally:
        db.close()
    assert os.path.exists(path)

    assert client.delete(f"/api/v1/knowledge/{kb}", headers=H).status_code == 200
    assert not os.path.exists(path), "知识库删了，磁盘上的原文件也该跟着没"


def test_delete_kb_also_clears_orphan_files(client):
    """目录里的**孤儿文件**（回滚 / 覆盖留下、已无 Document 行指向）也要一起清掉。

    路径约定是 `uploaded_files/<owner>/<kb>/`，该目录下的文件必然属于这个库，
    所以整目录清是安全的（安全审查 F2 的第二半）。
    """
    import os

    from app.services import document_service
    from tests.helpers import register_and_kb

    H, uid, kb = register_and_kb(client, "kborph")
    d = document_service.kb_upload_dir(uid, kb)
    os.makedirs(d, exist_ok=True)
    orphan = os.path.join(d, "orphan.bin")
    with open(orphan, "wb") as f:
        f.write(b"x")

    assert client.delete(f"/api/v1/knowledge/{kb}", headers=H).status_code == 200
    assert not os.path.exists(orphan), "没有 Document 行指向的孤儿文件也该清掉"


# ---------- F3：删会话（Postgres 下会被外键拦住） ----------


def test_delete_session_with_feedback_succeeds(client, auth_headers):
    """给消息点过赞也要能删掉会话。

    `feedback.message_id` 有外键指向 `chat_messages`，而 `query.delete()` 是 bulk delete、
    **不走 ORM 级联** —— 不清 feedback 的话 Postgres 会直接拒（安全审查 F3）。
    sqlite 以前不校验外键，所以这个洞在测试里看不出来。
    """
    import uuid

    from app.db.session import SessionLocal
    from app.models.entities import ChatMessage, ChatSession, Feedback, User

    H = auth_headers("sessdel")
    db = SessionLocal()
    try:
        uid = db.query(User).filter(User.username == "sessdel").first().id
        sid, mid = uuid.uuid4().hex, uuid.uuid4().hex
        db.add(ChatSession(id=sid, user_id=uid, kb_id="kb-x", title=""))
        db.commit()
        db.add(ChatMessage(id=mid, session_id=sid, role="assistant", content="答"))
        db.add(Feedback(message_id=mid, user_id=uid, rating=1, comment=""))
        db.commit()
    finally:
        db.close()

    assert client.delete(f"/api/v1/chat/sessions/{sid}", headers=H).status_code == 200

    db = SessionLocal()
    try:
        assert db.query(Feedback).filter(Feedback.message_id == mid).count() == 0, "反馈该一起删"
    finally:
        db.close()


# ---------- F4：kb / 会话归属 ----------


def test_debug_query_rejects_foreign_kb(client, auth_headers):
    """`/debug/query` 要和 `/chat/stream` 一样校验 kb 归属（安全审查 F4）。"""
    a = auth_headers("dbg_a")
    b = auth_headers("dbg_b")
    kb_a = client.post("/api/v1/knowledge", json={"name": "ka"}, headers=a).json()["id"]

    assert client.get("/api/v1/debug/query", params={"kb_id": kb_a, "question": "x"},
                      headers=b).status_code == 404
    # 本人访问正常（哪怕库里还没有文档）
    assert client.get("/api/v1/debug/query", params={"kb_id": kb_a, "question": "x"},
                      headers=a).status_code == 200


def test_session_ownership_is_asserted_in_service():
    """`_get_or_create_session` 自己也要断言属主。

    安全审查 F4：会话 id 由**客户端**提供，所以「拿别人的会话 id 读历史」是一条现成的路。
    别把安全性寄托在「每个调用方都记得先校验」上 —— 漏一个就是静默越权。
    """
    from app.db.session import SessionLocal
    from app.models.entities import ChatSession, User
    from app.services.chat_service import _get_or_create_session
    from app.utils.security import hash_password

    db = SessionLocal()
    try:
        db.add(User(id="u-a", username="svc_a", password_hash=hash_password("pw1234567890"), role="viewer"))
        db.add(User(id="u-b", username="svc_b", password_hash=hash_password("pw1234567890"), role="viewer"))
        db.commit()
        db.add(ChatSession(id="sess-own", user_id="u-a", kb_id="kb", title=""))
        db.commit()

        other = db.query(User).filter(User.id == "u-b").first()
        with pytest.raises(PermissionError):
            _get_or_create_session(db, other, "kb", "sess-own")
    finally:
        db.close()


# ---------- F5：注册 / 问答限流 ----------


def test_register_is_rate_limited(client, monkeypatch):
    """注册不鉴权，不限流就是「开放建号」—— 每个号都能烧服务端的 LLM 额度（安全审查 F5）。"""
    from app.api.v1 import auth as auth_mod
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "register_rate_limit_per_hour", 2, raising=False)
    auth_mod._register_limiter.clear()

    codes = [client.post("/api/v1/auth/register",
                         json={"username": f"rl{i}", "password": "pw1234567890"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


def test_chat_is_rate_limited_per_user(client, auth_headers, monkeypatch):
    """并发上限管的是「同时几个流」，管不住「一个接一个地发」—— 后者才烧钱（安全审查 F5）。"""
    from app.api.v1 import chat as chat_mod
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "chat_rate_limit_per_min", 2, raising=False)
    H = auth_headers("chatrl")
    kb = client.post("/api/v1/knowledge", json={"name": "k"}, headers=H).json()["id"]
    chat_mod._chat_limiter.clear()

    codes = [client.post("/api/v1/chat/stream", headers=H,
                         json={"kb_id": kb, "question": "q", "stream": True}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------- F9：sqlite 外键与生产一致 ----------


def test_sqlite_enforces_foreign_keys():
    """sqlite 默认**不校验**外键，而生产 Postgres 会 —— 两边不一致正是 F3 那种
    「测试全绿、上线 500」的成因。安全审查 F9 在连接时打开 `PRAGMA foreign_keys=ON`。
    """
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from app.db.session import SessionLocal
    from app.models.entities import ChatMessage

    db = SessionLocal()
    try:
        assert db.execute(text("PRAGMA foreign_keys")).scalar() == 1
        db.add(ChatMessage(id="orphan-msg", session_id="no-such-session", role="user", content="x"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
    finally:
        db.close()


# ---------- F14：优雅关停 ----------


def test_wait_for_processing_is_clean_when_idle():
    """没有在跑的索引线程时，关停等待应立即返回 0 —— 别让每次关停都白等 20 秒。"""
    from app.services import document_service

    assert document_service.wait_for_processing(timeout=0.1) == 0
