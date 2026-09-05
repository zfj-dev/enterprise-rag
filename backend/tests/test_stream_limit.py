"""SSE 流式每用户并发守卫 _StreamGuard 单测。"""
from app.api.v1.chat import _StreamGuard


def test_acquire_release_and_limit():
    g = _StreamGuard(limit=2)
    assert g.try_acquire("u1") is True
    assert g.try_acquire("u1") is True
    assert g.try_acquire("u1") is False  # 超限拒绝
    g.release("u1")
    assert g.try_acquire("u1") is True  # 释放后可再进
    # 多释放不报错、不越界
    g.release("u1")
    g.release("u1")
    g.release("u1")
    assert g._counts.get("u1", 0) == 0


def test_per_user_independent():
    g = _StreamGuard(limit=1)
    assert g.try_acquire("a") is True
    assert g.try_acquire("a") is False
    assert g.try_acquire("b") is True  # 其他用户不受影响
    assert g._counts["a"] == 1
    assert g._counts["b"] == 1
