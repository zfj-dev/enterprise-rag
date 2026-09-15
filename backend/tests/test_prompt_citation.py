from app.core.prompt import build_prompt, build_rewrite_prompt, format_context
from app.core.citation import validate_sources, apply_no_source_no_claim


def test_build_prompt_contains_parts():
    p = build_prompt("问题内容", "上下文内容")
    assert "问题内容" in p
    assert "上下文内容" in p
    assert "参考资料" in p


def test_build_rewrite_prompt():
    p = build_rewrite_prompt("它多少钱", [{"user": "电池价格", "assistant": "10元"}])
    assert "它多少钱" in p
    assert "电池价格" in p


def test_format_context_annotates_source():
    c = [{"chunk_id": "a1", "content": "内容", "metadata": {"doc_name": "手册", "page_num": 3}}]
    s = format_context(c)
    assert "手册" in s and "chunk_id: a1" in s


def test_validate_sources_empty():
    r = validate_sources([])
    assert not r.has_sources and r.coverage == 0.0


def test_validate_sources_usable():
    r = validate_sources([{"chunk_id": "a", "content": "x", "metadata": {}}])
    assert r.has_sources and r.stable_ids


def test_no_source_no_claim():
    assert "无法确定" in apply_no_source_no_claim("答案", validate_sources([]))
    keep = apply_no_source_no_claim("答案", validate_sources([{"chunk_id": "a", "content": "x", "metadata": {}}]))
    assert keep == "答案"


class _FakeVerifyLLM:
    def stream(self, messages):
        yield '{"claims":[{"claim":"比亚迪2025年营业收入为803.96亿元","supported":true},{"claim":"这是不支持的论断","supported":false}]}'


def test_verify_claims():
    from app.core.citation import verify_claims
    sources = [{"chunk_id": "a", "doc_name": "eval.txt", "page": 1, "text": "比亚迪2025年营业收入803.96亿元。"}]
    res = verify_claims("比亚迪2025年营业收入为803.96亿元。这是不支持的论断。", sources, _FakeVerifyLLM())
    assert res["total"] == 2
    assert res["supported"] == 1
    assert res["coverage"] == 0.5


def test_verify_claims_no_sources():
    from app.core.citation import verify_claims
    res = verify_claims("随便一句。", [], _FakeVerifyLLM())
    assert res["coverage"] == 0.0
    assert res["total"] == 1


# ---------- 覆盖率守门（票 B）----------

class _BoomVerifyLLM:
    def stream(self, messages):
        raise RuntimeError("网络挂了")


class _GarbageVerifyLLM:
    def stream(self, messages):
        yield "这不是 JSON"


SOURCES = [{"chunk_id": "a", "doc_name": "eval.txt", "page": 1,
            "text": "比亚迪2025年营业收入803.96亿元。"}]


def test_a_verifier_that_fails_is_unverified_not_one_hundred_percent():
    """校验没跑成 -> verified=False、coverage=None。

    以前这里回落成 **1.0**（「有来源就假定支撑」）—— 那是在编一个 100%：上游会据此
    以为有依据，评测也会把这个假的 100% 平均进覆盖率。防线建在这种数字上必塌（票 B）。
    """
    from app.core.citation import verify_claims
    res = verify_claims("比亚迪2025年营业收入为803.96亿元。", SOURCES, _BoomVerifyLLM())

    assert res["verified"] is False
    assert res["coverage"] is None
    assert "网络挂了" in res["note"]


def test_an_unparseable_verifier_reply_is_unverified_not_zero():
    """回复解析不了也是**未知**。

    回落成 0.0 等于说「一句依据都没有」，真值是「压根没判成」—— 拿它去拒答，
    和拿未知当结论是一回事。
    """
    from app.core.citation import verify_claims
    res = verify_claims("比亚迪2025年营业收入为803.96亿元。", SOURCES, _GarbageVerifyLLM())

    assert res["verified"] is False and res["coverage"] is None


class _ParaphrasingVerifyLLM:
    def stream(self, messages):
        yield '{"claims":[{"claim":"裁判自己缩写了这句话","supported":false}]}'


def test_a_verifier_that_paraphrases_everything_is_unverified_not_zero():
    """裁判回的论断跟发出去的一条都对不上 -> 是**对不上**，不是「一条都没支撑」。

    以前这只影响一个指标；现在 coverage 直接决定要不要拒答，认错方向就会把好答案拦掉。
    """
    from app.core.citation import verify_claims
    res = verify_claims("比亚迪2025年营业收入为803.96亿元。", SOURCES,
                        _ParaphrasingVerifyLLM())

    assert res["verified"] is False and res["coverage"] is None


def test_no_sources_is_a_real_zero_not_an_unknown():
    """没来源是**确定的** 0（支撑数为 0），不是「没校验成」—— 两者不许混。"""
    from app.core.citation import verify_claims
    res = verify_claims("随便一句。", [], _FakeVerifyLLM())

    assert res["verified"] is True and res["coverage"] == 0.0


def test_the_coverage_guard_refuses_an_answer_with_no_support():
    """有论断、但一条依据都没找到 -> 改口拒答，且话术要说清是「没依据」而不是「没检索到」。"""
    from app.core.citation import apply_coverage_guard, coverage_too_low
    v = {"coverage": 0.0, "total": 2, "supported": 0, "verified": True}

    assert coverage_too_low(v, 0.0)
    out = apply_coverage_guard("据我所知，答案是 42。", v, 0.0)

    assert "无法确定" in out and "42" not in out
    assert "未检索到" not in out          # 检索到了东西，是依据不成立


def test_the_coverage_guard_lets_a_supported_answer_through():
    from app.core.citation import apply_coverage_guard, coverage_too_low
    v = {"coverage": 0.5, "total": 2, "supported": 1, "verified": True}

    assert not coverage_too_low(v, 0.0)
    assert apply_coverage_guard("答案是 42。", v, 0.0) == "答案是 42。"
    assert coverage_too_low(v, 0.5)                   # 门槛调到 0.5 就该拦


def test_an_unverified_answer_is_never_blocked():
    """**没校验成 ≠ 没依据**：拿未知去否掉一个本来正确的回答，同样是假结论（票 B）。"""
    from app.core.citation import apply_coverage_guard, coverage_too_low
    v = {"coverage": None, "total": 2, "supported": None, "verified": False}

    assert not coverage_too_low(v, 0.0)
    assert not coverage_too_low(v, 0.9)
    assert apply_coverage_guard("答案是 42。", v, 0.9) == "答案是 42。"
