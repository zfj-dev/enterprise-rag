"""RAGAS 四项裁判（票 05 / #12）：按官方定义算；LLM 与嵌入都注入 stub —— 无网络、确定。"""
from __future__ import annotations

import pytest

from app.eval_judge import JudgeUnavailable, RagasJudge


class ScriptedLLM:
    """按提示词里的标记回脚本化答案。api_key 非空，好过掉「没配 Key」那道检查。"""

    is_fake = False
    api_key = "k"
    model = "qwen-turbo"

    def __init__(self, replies: dict):
        self.replies = replies

    def stream(self, messages):
        prompt = messages[-1]["content"]
        for marker, reply in self.replies.items():
            if marker in prompt:
                yield reply
                return
        raise AssertionError("未脚本化的提示词：%s" % prompt[:80])


class StubEmbedding:
    """文本含「相关」给 [1,0]，否则 [0,1] —— 相似度只可能是 1 或 0。"""

    def encode(self, texts):
        return [[1.0, 0.0] if "相关" in t else [0.0, 1.0] for t in texts]


def _judge(replies, **kw):
    return RagasJudge(ScriptedLLM(replies), StubEmbedding(), **kw)


# ---------- 四项各自的算法 ----------

def test_faithfulness_is_supported_claims_over_total_claims():
    replies = {
        "拆成若干条": "甲成立\n乙成立\n丙不成立",
        "逐条判断": "[true, true, false]",
        "反推出": "一问",
        "对回答": "[true]",
    }
    got = _judge(replies)("问", "答案", [{"text": "上下文"}])
    assert got["faithfulness"] == pytest.approx(2 / 3)


def test_answer_relevancy_averages_similarity_to_generated_questions():
    replies = {
        "反推出": "原问题相关\n完全无关的另一问",
        "拆成若干条": "一条论断",
        "逐条判断": "[true]",
        "对回答": "[true]",
    }
    got = _judge(replies)("原问题相关", "答案", [{"text": "上下文"}])
    assert got["answer_relevancy"] == pytest.approx(0.5)      # 一个相似度 1、一个 0


def test_context_precision_follows_the_average_precision_formula():
    """有用的块排得越靠前分越高：命中在 1、3 位 → (1/1 + 2/3) / 2。"""
    replies = {
        "对回答": "[true, false, true]",
        "拆成若干条": "一条论断",
        "逐条判断": "[true]",
        "反推出": "一问",
    }
    got = _judge(replies)("问", "答", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
    assert got["context_precision"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_context_precision_is_zero_when_nothing_is_useful():
    replies = {"对回答": "[false, false]", "拆成若干条": "一条", "逐条判断": "[true]", "反推出": "一问"}
    assert _judge(replies)("问", "答", [{"text": "a"}, {"text": "b"}])["context_precision"] == 0.0


def test_context_recall_uses_the_reference_answer():
    replies = {"拆成若干条": "论点一\n论点二", "逐条判断": "[true, false]",
               "反推出": "一问", "对回答": "[true]"}
    got = _judge(replies)("问", "答", [{"text": "ctx"}], reference="参考答案")
    assert got["context_recall"] == pytest.approx(0.5)


def test_context_recall_is_zero_without_a_reference():
    replies = {"拆成若干条": "一条", "逐条判断": "[true]", "反推出": "一问", "对回答": "[true]"}
    got = _judge(replies)("问", "答", [{"text": "ctx"}])
    assert got["context_recall"] == 0.0


# ---------- 裁判口径与失败路径 ----------

def test_label_records_the_fixed_judge_caliber():
    label = _judge({}).label
    assert "qwen-turbo" in label and "temp=0" in label


def test_missing_api_key_raises_before_any_call():
    class NoKey:
        api_key = ""

    with pytest.raises(JudgeUnavailable):
        RagasJudge(NoKey(), StubEmbedding())


def test_unparseable_judge_reply_raises_instead_of_faking_a_number():
    replies = {"拆成若干条": "甲\n乙", "逐条判断": "我觉得都还行"}      # 不是 JSON 数组
    with pytest.raises(JudgeUnavailable):
        _judge(replies)("问", "答", [{"text": "ctx"}])


def test_wrong_length_bool_array_raises():
    replies = {"拆成若干条": "甲\n乙", "逐条判断": "[true]"}          # 两条论断只给一个
    with pytest.raises(JudgeUnavailable):
        _judge(replies)("问", "答", [{"text": "ctx"}])


def test_llm_failure_becomes_judge_unavailable():
    class Boom:
        api_key = "k"

        def stream(self, messages):
            raise RuntimeError("网络断了")
            yield  # pragma: no cover

    with pytest.raises(JudgeUnavailable) as e:
        RagasJudge(Boom(), StubEmbedding())("问", "答", [{"text": "ctx"}])
    assert "网络断了" in str(e.value)


def test_sources_without_text_are_ignored_not_crashed():
    replies = {"拆成若干条": "一条", "逐条判断": "[true]", "反推出": "一问", "对回答": "[true]"}
    sources = [{"page": 1}, {"text": "   "}, {"text": "真上下文"}]
    got = _judge(replies)("问", "答", sources)                 # 只有 1 段真上下文
    assert got["context_precision"] == pytest.approx(1.0)


def test_the_judge_returns_exactly_the_canonical_metric_names():
    """裁判回的键必须与核心汇总的那四个指标名一一对应 —— 少一个就等于没算。"""
    from app.eval_core import RAGAS_METRICS

    replies = {"拆成若干条": "一条", "逐条判断": "[true]", "反推出": "一问", "对回答": "[true]"}
    got = _judge(replies)("问", "答", [{"text": "ctx"}])
    assert set(got) == set(RAGAS_METRICS)
