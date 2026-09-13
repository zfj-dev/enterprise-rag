"""RAGAS 四项裁判（票 05 / #12；布尔判定改「一条一问」见 #53）：LLM 与嵌入都注入 stub —— 无网络、确定。

**协议变了**：早先是一次问 N 条、要求模型回**长度恰好为 N** 的数组；真机上 10 条里 9 条因为
「回了 5 个、要 6 个」被整条丢掉（#53）。现在**一条一问**，n 恒为 1。
"""
from __future__ import annotations

import pytest

from app.eval_core import RAGAS_METRICS
from app.eval_judge import JudgeUnavailable, RagasJudge, parse_bool


class ScriptedLLM:
    """按提示词里的标记回脚本化答案。

    值给**列表**时按调用次序逐个弹出 —— 逐条判定的用例要的就是「同一标记、多次调用、答案不同」。
    """

    is_fake = False
    api_key = "k"
    model = "qwen-turbo"

    def __init__(self, replies: dict):
        self.replies = {k: (list(v) if isinstance(v, list) else v) for k, v in replies.items()}
        self.prompts: list[str] = []

    def stream(self, messages):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        for marker, reply in self.replies.items():
            if marker in prompt:
                if isinstance(reply, list):
                    if not reply:
                        raise AssertionError("脚本已用完（又调了一次）：%s" % prompt[:60])
                    yield reply.pop(0)
                    return
                yield reply
                return
        raise AssertionError("未脚本化的提示词：%s" % prompt[:80])


class StubEmbedding:
    """文本含「相关」给 [1,0]，否则 [0,1] —— 相似度只可能是 1 或 0。"""

    def encode(self, texts):
        return [[1.0, 0.0] if "相关" in t else [0.0, 1.0] for t in texts]


def _judge(replies, **kw):
    return RagasJudge(ScriptedLLM(replies), StubEmbedding(), **kw)


SPLIT = "拆成若干条"
SUPPORT = "能否**仅凭**"
CTX_REL = "有没有用"
GENQ = "反推出"


# ---------- 四项各自的算法 ----------

def test_faithfulness_is_supported_claims_over_total_claims():
    replies = {
        SPLIT: "甲成立\n乙成立\n丙不成立",
        SUPPORT: ["true", "true", "false"],       # 一条一问，按次序弹
        GENQ: "一问",
        CTX_REL: ["true"],
    }
    got = _judge(replies)("问", "答案", [{"text": "上下文"}])
    assert got["faithfulness"] == pytest.approx(2 / 3)


def test_answer_relevancy_averages_similarity_to_generated_questions():
    replies = {
        GENQ: "原问题相关\n完全无关的另一问",
        SPLIT: "一条论断",
        SUPPORT: ["true"],
        CTX_REL: ["true"],
    }
    got = _judge(replies)("原问题相关", "答案", [{"text": "上下文"}])
    assert got["answer_relevancy"] == pytest.approx(0.5)      # 一个相似度 1、一个 0


def test_context_precision_follows_the_average_precision_formula():
    """有用的块排得越靠前分越高：命中在 1、3 位 → (1/1 + 2/3) / 2。"""
    replies = {CTX_REL: ["true", "false", "true"], SPLIT: "一条论断",
               SUPPORT: ["true"], GENQ: "一问"}
    got = _judge(replies)("问", "答", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
    assert got["context_precision"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_context_precision_is_zero_when_nothing_is_useful():
    replies = {CTX_REL: ["false", "false"], SPLIT: "一条", SUPPORT: ["true"], GENQ: "一问"}
    assert _judge(replies)("问", "答", [{"text": "a"}, {"text": "b"}])["context_precision"] == 0.0


def test_context_recall_uses_the_reference_answer():
    # 拆论断会被调**两次**（答案一次、参考答案一次）—— 逐条判定之后这里得把两次的份都备足
    replies = {SPLIT: ["一条", "论点一\n论点二"], SUPPORT: ["true", "true", "false"],
               GENQ: "一问", CTX_REL: ["true"]}
    got = _judge(replies)("问", "答", [{"text": "ctx"}], reference="参考答案")
    assert got["context_recall"] == pytest.approx(0.5)


def test_context_recall_is_zero_without_a_reference():
    replies = {SPLIT: "一条", SUPPORT: ["true"], GENQ: "一问", CTX_REL: ["true"]}
    assert _judge(replies)("问", "答", [{"text": "ctx"}])["context_recall"] == 0.0


# ---------- 一条一问：把「数数」从协议里拿掉（#53）----------

def test_each_claim_gets_its_own_call_so_the_count_can_never_mismatch():
    """**这一票的核心**：早先「一次问 N 条、要求回长度恰好 N」，模型数错就整条丢掉。

    真机上 10 条里 9 条这么没的。一条一问之后 n 恒为 1 —— 模型数不数得对**无关紧要**。
    """
    llm = ScriptedLLM({SPLIT: "甲\n乙\n丙", SUPPORT: ["true", "false", "true"],
                       GENQ: "一问", CTX_REL: ["true"]})
    RagasJudge(llm, StubEmbedding())("问", "答", [{"text": "ctx"}])

    # 三条论断 = 三次判定调用（不再是「一次要一个三元数组」）
    assert sum(1 for p in llm.prompts if SUPPORT in p) == 3


def test_a_single_unreadable_verdict_is_reported_not_guessed():
    """某一条认不出来时，报的是**不可用**（绝不猜），而不是把它悄悄算成 false。"""
    replies = {SPLIT: "甲\n乙", SUPPORT: ["true", "我拿不准"], GENQ: "一问", CTX_REL: ["true"]}
    with pytest.raises(JudgeUnavailable):
        _judge(replies)("问", "答", [{"text": "ctx"}])


# ---------- 布尔解析：中文里「先判否定」----------

@pytest.mark.parametrize("raw", ["true", "yes", "是", "可以", "能", "支持", "相关"])
def test_affirmative_replies_parse_true(raw):
    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["false", "no", "不是", "否", "不能", "不可以", "不支持", "不相关"])
def test_negative_replies_parse_false(raw):
    """「不可以」里含「可以」、「不支持」里含「支持」—— 关键词匹配会把「否」判成「是」，
    所以这里认的是**整条回复**，不是「句子里有没有某个词」。"""
    assert parse_bool(raw) is False


@pytest.mark.parametrize("raw", ["", "  ", "我觉得都还行", "说不清",
                              "无法推出", "能推出来", "不确定是否相关", "not true"])   # 多话 = 没答，不猜
def test_unrecognisable_replies_parse_to_none(raw):
    """**绝不猜**：只有「本身就是判断」的短回复才认。

    早先的版本在整句里找关键词，于是「不确定是否相关」被判成 false、「not true」被判成 true ——
    都是编出来的结论。
    """
    assert parse_bool(raw) is None


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
    replies = {SPLIT: "甲", SUPPORT: "我觉得都还行", GENQ: "一问", CTX_REL: ["true"]}
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
    replies = {SPLIT: "一条", SUPPORT: ["true"], GENQ: "一问", CTX_REL: ["true"]}
    sources = [{"page": 1}, {"text": "   "}, {"text": "真上下文"}]
    got = _judge(replies)("问", "答", sources)                 # 只有 1 段真上下文
    assert got["context_precision"] == pytest.approx(1.0)


def test_the_judge_returns_exactly_the_canonical_metric_names():
    """裁判回的键必须与核心汇总的那四个指标名一一对应 —— 少一个就等于没算。"""
    replies = {SPLIT: "一条", SUPPORT: ["true"], GENQ: "一问", CTX_REL: ["true"]}
    got = _judge(replies)("问", "答", [{"text": "ctx"}])
    assert set(got) == set(RAGAS_METRICS)
