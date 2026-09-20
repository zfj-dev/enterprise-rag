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


def test_seed_admin_real_mode_never_uses_public_password(monkeypatch):
    """真实模式没配 ADMIN_PASSWORD → 随机生成，绝不回落 admin123。"""
    import app.main as m
    from app.db.session import SessionLocal
    from app.models.entities import User
    from app.utils.security import verify_password

    class _S:
        use_real = True
        admin_password = ""

    monkeypatch.setattr(m, "get_settings", lambda: _S())

    m._seed_admin()

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "admin").first()
        assert u is not None
        assert not verify_password("admin123", u.password_hash)
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
    """新口令的下限比注册更严（12 位）—— 这是要长期用下去的那个。"""
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
