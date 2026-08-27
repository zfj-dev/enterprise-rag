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
