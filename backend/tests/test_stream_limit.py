"""SSE 流式每用户并发守卫 _StreamGuard 单测。"""
from app.api.v1.chat import _StreamGuard


def test_acquire_hands_out_a_token_and_release_frees_exactly_it():
    g = _StreamGuard(limit=2)
    t1 = g.try_acquire("u1")
    t2 = g.try_acquire("u1")
    assert t1 and t2 and t1 != t2
    assert g.try_acquire("u1") is None       # 超限拒绝
    g.release("u1", t1)
    assert g.try_acquire("u1")                # 释放后可再进
    # 同一个令牌释放两次不报错、也不越界
    g.release("u1", t1)
    g.release("u1", t1)
    assert g.try_acquire("u1") is None


def test_releasing_twice_does_not_free_someone_elses_slot():
    """断连时名额有两条释放路径（生成器收尾 + background）—— 重复释放不许把别人的名额放掉。

    按用户减计数就会踩这个：2 个流在跑，其中一个释放两次，计数从 2 掉到 0，另一个的名额没了。
    """
    g = _StreamGuard(limit=1)
    t = g.try_acquire("u1")
    assert t
    g.release("u1", t)
    g.release("u1", t)                        # 同一条流的两条释放路径都跑到了
    assert g.try_acquire("u1")                # 名额回到 1 个 —— 没有被多放
    assert g.try_acquire("u1") is None        # 也只能有 1 个


def test_per_user_independent():
    g = _StreamGuard(limit=1)
    ta = g.try_acquire("a")
    assert ta and g.try_acquire("a") is None
    tb = g.try_acquire("b")                   # 其他用户不受影响
    assert tb
    assert set(g._held) == {"a", "b"}


def test_a_stale_token_from_another_user_frees_nothing():
    """令牌是别人发的 —— 不能动我的名额。"""
    g = _StreamGuard(limit=1)
    assert g.try_acquire("a")
    g.release("a", "别人的令牌")
    assert g.try_acquire("a") is None
