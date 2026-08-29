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
