"""生产化相关单测:登录限流、语义缓存 Redis 后端(且降级内存)。"""
import uuid

from app.api.v1.auth import _login_attempts
from app.config import get_settings
from app.core.cache import SemanticCache
from app.core.embedding import FakeEmbedding


def test_login_rate_limit_429(client):
    """同一用户名 60s 内超过 login_rate_limit_per_min 次 -> 429。"""
    username = "rl_" + uuid.uuid4().hex
    _login_attempts.pop(username, None)   # 清掉该用户名计数,避免跨测试干扰
    limit = get_settings().login_rate_limit_per_min
    codes = [client.post("/api/v1/auth/login", json={"username": username, "password": "x"}).status_code
             for _ in range(limit)]
    assert all(c == 401 for c in codes)   # 前 limit 次:用户名不存在 -> 401
    after = client.post("/api/v1/auth/login", json={"username": username, "password": "x"})
    assert after.status_code == 429


def test_semantic_cache_redis_backend_falls_back_and_roundtrips():
    """backend='redis' 但无 redis server 时降级内存;get/put 相似命中。"""
    emb = FakeEmbedding(dim=64)
    c = SemanticCache(embedding=emb, backend="redis")   # 无 redis -> _init_redis 捕获 -> _redis=None
    q = "比亚迪营收多少"
    c.put(q, "kb1", "803.96亿")
    hit = c.get(q, "kb1")
    assert hit and hit["answer"] == "803.96亿"
    assert c.get(q, "kb2") is None   # 其它库不命中


def test_semantic_cache_threshold_miss():
    """相似度低于阈值(0.99)时不应命中。"""
    emb = FakeEmbedding(dim=64)
    c = SemanticCache(embedding=emb, backend="memory", threshold=0.99)
    c.put("a", "kb1", "ans1")
    assert c.get("完全不相干的问题xyz", "kb1") is None
