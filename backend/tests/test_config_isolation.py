"""测试必须与本机配置隔离（否则 pre-commit 会被本机的 backend/.env 卡住）。

开发者本机的 `.env` 是真实模式（USE_REAL=true + 托管嵌入 + 真 Key），而**环境变量优先于 .env**。
conftest 在导入 app 之前把档位钉回演示/测试档 —— 这几条就是钉子的回归用例：
哪天有人「顺手清理」了那段，这里会红。
"""
from __future__ import annotations

from app.config import get_settings


def test_tests_never_run_in_real_mode():
    s = get_settings()
    assert s.use_real is False


def test_tests_never_pick_the_hosted_providers():
    s = get_settings()
    assert s.embedding_provider == "fake"
    assert s.reranker_provider == "fake"
    assert s.llm_provider == "fake"


def test_tests_never_carry_a_real_api_key():
    s = get_settings()
    assert not s.llm_api_key and not s.embedding_api_key and not s.rerank_api_key


def test_tests_use_the_default_concurrency():
    """本机 .env 把它调成 8 了；测试要按默认 2 跑，否则并发上限相关的用例会失真。"""
    assert get_settings().max_concurrent_streams_per_user == 2
