"""测试配置：隔离的临时 sqlite 库 + 每个测试重置运行时（干净的 in-memory 向量库/BM25）。"""
import os
import tempfile

_D = tempfile.mkdtemp(prefix="rag_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_D}/test_rag.db"
os.environ["SECRET_KEY"] = "test-secret"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import app
    from app.db.session import engine
    from app.models.entities import Base

    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_state():
    """每个测试前：清空数据库 + 重置运行时单例（向量库/BM25 归零）。"""
    from app.db.session import engine
    from app.models.entities import Base

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    import app.api.deps as deps

    deps._runtime = None
    yield


# ---- Auth 专用 fixtures ----
@pytest.fixture
def make_user(client):
    """工厂：直插指定 role 的用户并返回其 access_token。

    说明：注册接口 POST /auth/register 永远把 role 硬编码成 "viewer"，且 body 无 role 字段；
    所以要用 uploader/admin 角色，只能绕过 API 直插数据库（本 fixture 所用）。同时兼顾测试登录。
    """
    from app.db.session import SessionLocal
    from app.models.entities import User
    from app.utils.security import hash_password

    def _make(username: str, role: str = "viewer", password: str = "pw123456") -> str:
        db = SessionLocal()
        try:
            if not db.query(User).filter(User.username == username).first():
                db.add(User(username=username, password_hash=hash_password(password), role=role))
                db.commit()
        finally:
            db.close()
        r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return r.json()["access_token"]

    return _make


@pytest.fixture
def auth_headers(make_user):
    """工厂：返回指定用户（默认 viewer）的 Authorization 头。"""
    def _h(username: str, role: str = "viewer", password: str = "pw123456") -> dict:
        tok = make_user(username, role, password)
        return {"Authorization": f"Bearer {tok}"}

    return _h
