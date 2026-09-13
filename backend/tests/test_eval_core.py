"""评测核心（票 01 / #8）：注入式纯函数 —— 不连服务、不调 LLM。

被评的问答函数与裁判函数都注入 stub，因此断言是确定的、离线的。
"""
from __future__ import annotations

import pytest

from app.eval_core import is_refusal, normalize, run_eval

GOLDEN = [
    {"question": "Q1", "expect": "803.96"},
    {"question": "Q2", "expect": "PyCharm", "page": 29},
    {"question": "Q3", "expect": "RTX 3060", "page": 31},
]

ANSWERS = {
    "Q1": {"answer": "803.96 亿元 [来源: a.pdf, 第1页]",
           "sources": [{"text": "营业收入 803.96 亿元", "page": 1}]},
    "Q2": {"answer": "用的 PyCharm [来源: b.pdf, 第29页]",
           "sources": [{"text": "PyCharm", "page": 29}]},
    "Q3": {"answer": "不知道", "sources": [{"text": "与本问无关", "page": 1}]},
}


def _answer_fn(table):
    return lambda question: table[question]


# ---------- 指标计算 ----------

def test_fact_grounded_and_page_rates():
    rep = run_eval(GOLDEN, _answer_fn(ANSWERS))

    assert rep.total == 3
    assert [x.fact_hit for x in rep.items] == [True, True, False]      # Q3 答案里没有期望事实
    assert [x.grounded for x in rep.items] == [True, True, False]      # 且来源里也没有
    assert rep.fact_rate == pytest.approx(2 / 3)
    assert rep.grounded_rate == pytest.approx(2 / 3)
    assert rep.page_rate == pytest.approx(1 / 2)                       # 只有 Q2 页码对得上


def test_page_rate_is_none_when_no_item_declares_a_page():
    rep = run_eval([{"question": "Q1", "expect": "803.96"}], _answer_fn(ANSWERS))
    assert rep.page_rate is None
    assert rep.items[0].page_hit is None

    text = "\n".join(rep.to_lines())
    assert "引用页码正确 不适用" in text and "分母为 0" in text


def test_empty_goldenset_is_all_zero():
    rep = run_eval([], lambda q: {"answer": "", "sources": []})
    assert rep.total == 0
    assert rep.fact_rate == 0.0 and rep.grounded_rate == 0.0 and rep.page_rate is None


def test_missing_expect_is_a_miss_not_a_skip():
    """期望事实缺失时标为未命中，而不是静默跳过（否则分母会变小、数字虚高）。"""
    golden = [{"question": "Q", "expect": ""}, {"question": "R", "expect": "甲"}]
    table = {"Q": {"answer": "随便", "sources": []}, "R": {"answer": "甲", "sources": []}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.total == 2
    assert rep.items[0].fact_hit is False
    assert rep.fact_rate == pytest.approx(0.5)
    assert rep.missing_expect_count == 1

    text = "\n".join(rep.to_lines())
    assert "(未写期望事实)" in text and "按未命中计" in text


def test_answer_fn_may_omit_sources():
    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"})
    assert rep.items[0].fact_hit is True
    assert rep.items[0].grounded is False
    assert rep.items[0].pages == []


# ---------- 判据口径 ----------

def test_normalize_removes_whitespace_and_case():
    assert normalize("表 4 . 1") == normalize("表4.1")
    assert normalize("Windows  11") == normalize("windows11")
    assert normalize(None) == ""
    assert normalize("RTX 3060") == "rtx3060"


def test_parser_inserted_spaces_do_not_break_the_hit():
    """解析器会在数字/标点间插空格（'表 3 . 1'），判据必须照命中。"""
    golden = [{"question": "表3.1 的实验环境", "expect": "RTX 3060", "page": 29}]
    table = {"表3.1 的实验环境": {"answer": "实验用 RTX  3060 显卡",
                                  "sources": [{"text": "GPU：RTX 3060", "page": 29}]}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.items[0].fact_hit is True
    assert rep.items[0].grounded is True
    assert rep.items[0].page_hit is True


# ---------- 裁判缝 ----------

def test_judge_fn_is_called_per_item_and_recorded():
    seen = []

    def judge(question, answer, sources, reference):
        seen.append(question)
        return {"faithful": True}

    rep = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=judge)

    assert seen == ["Q1", "Q2", "Q3"]
    assert [x.judged for x in rep.items] == [{"faithful": True}] * 3


def test_judge_receives_the_answer_and_sources():
    got = {}

    def judge(question, answer, sources, reference):
        got["question"], got["answer"], got["sources"] = question, answer, sources
        return {}

    run_eval([GOLDEN[0]], _answer_fn(ANSWERS), judge_fn=judge)
    assert got["question"] == "Q1"
    assert got["answer"] == ANSWERS["Q1"]["answer"]
    assert got["sources"] == ANSWERS["Q1"]["sources"]


def test_no_judge_fn_leaves_judged_none():
    rep = run_eval(GOLDEN, _answer_fn(ANSWERS))
    assert all(x.judged is None for x in rep.items)


def test_stub_run_is_deterministic():
    """stub 注入下同一输入两次跑出完全一样的报告（无网络、无真实 LLM）。"""
    a = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=lambda q, a_, s: {"n": 1})
    b = run_eval(GOLDEN, _answer_fn(ANSWERS), judge_fn=lambda q, a_, s: {"n": 1})
    assert a.to_lines() == b.to_lines()


# ---------- 报告文本 ----------

def test_report_lines_state_the_criterion_and_the_numbers():
    text = "\n".join(run_eval(GOLDEN, _answer_fn(ANSWERS)).to_lines())

    assert "口径" in text and "去空白" in text          # 数字要有口径，才可信
    assert "67%" in text                                # 答案含期望事实 2/3
    assert "50%" in text                                # 引用页码正确 1/2
    assert "Q3" in text                                 # 逐条都在


# ---------- 缺失时的口径：不静默跳过 ----------

def test_item_without_declared_page_is_stated_not_silently_dropped():
    """没声明页码的条目不计入页码率 —— 但报告必须写明条数，不能看着像被跳过。"""
    golden = [{"question": "Q1", "expect": "甲"}, {"question": "Q2", "expect": "乙", "page": 5}]
    table = {"Q1": {"answer": "甲", "sources": [{"text": "甲", "page": 1}]},
             "Q2": {"answer": "乙", "sources": [{"text": "乙", "page": 5}]}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.total == 2
    assert rep.undeclared_page_count == 1
    assert rep.page_rate == 1.0                       # 分母只算声明了页码的那条
    text = "\n".join(rep.to_lines())
    assert "另有 1 条黄金集条目未声明页码" in text
    assert "未声明页码" in text                        # 逐条那行也点名了
def test_item_whose_run_produced_nothing_is_a_miss_not_a_skip():
    """答案空、来源空：声明了期望与页码的那条要判成未命中，而不是消失。"""
    rep = run_eval([{"question": "Q", "expect": "甲", "page": 3}],
                   lambda q: {"answer": "", "sources": []})

    assert rep.total == 1
    assert rep.items[0].fact_hit is False
    assert rep.items[0].grounded is False
    assert rep.items[0].page_hit is False             # 声明了页码却没有来源页码
    assert rep.page_rate == 0.0


def test_all_three_numbers_always_get_a_line():
    """三个数字必须行行都在 —— 空黄金集也不例外。"""
    text = "\n".join(run_eval([], lambda q: {}).to_lines())
    assert "答案含期望事实" in text
    assert "引用忠实度" in text
    assert "引用页码正确" in text

# ---------- 拒答判据（免 LLM） ----------

def test_is_refusal_matches_explicit_refusals():
    assert is_refusal("根据现有资料无法确定（未检索到可引用的内容）。") is True
    assert is_refusal("资料中没有相关信息，无法回答。") is True
    assert is_refusal("未找到相关内容。") is True


def test_is_refusal_covers_the_systems_own_phrasings():
    """系统自己的拒答话术必须全认出来 —— 兜底串与 prompt 规则 2/3 的敏感信息拒答。"""
    from app.core.citation import apply_no_source_no_claim, validate_sources

    assert is_refusal(apply_no_source_no_claim("原始答案", validate_sources([]))) is True
    assert is_refusal("抱歉，涉及薪资等敏感信息，我不能提供。") is True
    assert is_refusal("文中未提及该数据。") is True
    assert is_refusal("该文档未包含相关内容。") is True


def test_long_answer_that_merely_mentions_refusal_is_not_a_refusal():
    """先拒后硬答：长篇里夹一句拒答，仍是在拿模型自身知识作答 —— 算成拒答会把数字抬高。"""
    hard = "资料中没有相关信息。" + "不过据我所知，" + "这个数字大约是 42。" * 20
    assert is_refusal(hard) is False
    assert is_refusal("资料中没有相关信息。") is True     # 同一句话，短的就是真拒答


def test_is_refusal_does_not_fire_on_ordinary_answers():
    assert is_refusal("比亚迪2025年营业收入为 803.96 亿元。") is False
    assert is_refusal("这个数字不太确定，但资料显示约为 803.96 亿元。") is False   # 有"不确定"但不是拒答
    assert is_refusal("") is False                                              # 空答案 = 没答，不是拒答


# ---------- 负样本与拒答率 ----------

NEG = [{"question": "公司的考勤打卡截止时间是几点", "negative": True},
       {"question": "软件著作权登记号是多少", "negative": True}]


def test_negative_samples_are_supported_and_refusal_rate_is_computed():
    golden = [{"question": "Q1", "expect": "803.96"}] + NEG
    table = {
        "Q1": {"answer": "803.96 [来源: a.pdf, 第1页]", "sources": [{"text": "803.96", "page": 1}]},
        "公司的考勤打卡截止时间是几点": {"answer": "根据现有资料无法确定。", "sources": []},
        "软件著作权登记号是多少": {"answer": "登记号是 2024SR1234567。", "sources": []},
    }
    rep = run_eval(golden, _answer_fn(table))

    assert rep.total == 3
    assert len(rep.positives) == 1 and len(rep.negatives) == 2
    assert rep.refuse_rate == pytest.approx(0.5)              # 一个明确拒答、一个硬答
    assert [x.refused for x in rep.negatives] == [True, False]
    assert rep.refused_count == 1


def test_negatives_stay_out_of_the_fact_and_page_denominators():
    """负样本没有期望事实，不能被当成"未命中"把事实命中率拉低。"""
    golden = [{"question": "Q1", "expect": "甲"}] + NEG
    table = {"Q1": {"answer": "甲", "sources": []},
             "公司的考勤打卡截止时间是几点": {"answer": "无法确定", "sources": []},
             "软件著作权登记号是多少": {"answer": "无法确定", "sources": []}}
    rep = run_eval(golden, _answer_fn(table))

    assert rep.fact_rate == 1.0            # 分母只有那 1 条正样本
    assert rep.missing_expect_count == 0   # 负样本没写 expect，不算数据缺口
    assert rep.undeclared_page_count == 1  # 只数正样本


def test_report_puts_refusal_rate_next_to_fact_hit_rate():
    """两者并列 —— 免得"拒答率高是因为什么都不答"被误读。"""
    golden = [{"question": "Q1", "expect": "甲"}] + NEG
    table = {"Q1": {"answer": "甲", "sources": []},
             "公司的考勤打卡截止时间是几点": {"answer": "无法确定", "sources": []},
             "软件著作权登记号是多少": {"answer": "登记号是 2024SR1234567。", "sources": []}}
    text = "\n".join(run_eval(golden, _answer_fn(table)).to_lines())

    assert text.index("答案含期望事实") < text.index("拒答率") < text.index("引用忠实度")
    assert "另有 2 条负样本另计拒答率" in text
    assert "明确拒答 1，未拒答 1" in text
    assert "REFUSED" in text and "ANSWERED" in text          # 逐条区分两种结局
    assert "负样本(期望拒答)" in text


def test_refusal_rate_absent_when_there_are_no_negative_samples():
    rep = run_eval(GOLDEN, _answer_fn(ANSWERS))
    assert rep.refuse_rate is None
    assert "拒答率 不适用" in "\n".join(rep.to_lines())


# ---------- RAGAS 四项（票 05） ----------

def test_report_averages_the_four_ragas_metrics():
    calls = []

    def judge(question, answer, sources, reference):
        calls.append((question, reference))
        v = 0.8 if question == "Q1" else 0.4
        return {"faithfulness": v, "answer_relevancy": v, "context_precision": v, "context_recall": v}

    golden = [{"question": "Q1", "expect": "甲"}, {"question": "Q2", "expect": "乙"}]
    rep = run_eval(golden, _answer_fn({"Q1": {"answer": "甲"}, "Q2": {"answer": "乙"}}),
                   judge_fn=judge, judge_label="RAGAS(judge=stub, temp=0.0)")

    assert calls == [("Q1", "甲"), ("Q2", "乙")]        # 裁判拿得到参考答案（context_recall 要用）
    assert rep.ragas_count == 2
    assert set(rep.ragas) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert rep.ragas["faithfulness"] == pytest.approx(0.6)      # (0.8 + 0.4) / 2

    text = "\n".join(rep.to_lines())
    assert "=== RAGAS 四项 ===" in text
    assert "RAGAS(judge=stub, temp=0.0)" in text                # 口径必须写进报告
    assert "计入 2/2 条" in text


def test_judge_failure_is_recorded_not_swallowed():
    """裁判挂了要记下来并明说 —— 宁可报错，也不给看着正常的假数字。"""
    def boom(question, answer, sources, reference):
        raise RuntimeError("裁判超时")

    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"}, judge_fn=boom)

    assert rep.ragas is None
    assert "裁判超时" in (rep.judge_error or "")
    text = "\n".join(rep.to_lines())
    assert "裁判不可用" in text and "RuntimeError" in text
    assert "faithfulness" not in text          # 一个数都不许编


def test_no_judge_means_no_ragas_numbers():
    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"})
    assert rep.ragas is None and rep.ragas_count == 0
    text = "\n".join(rep.to_lines())
    assert "未接裁判" in text
    assert "faithfulness" not in text


def test_judge_giving_no_usable_scores_is_said_so():
    """裁判跑了但没给出可解析的分：不能报成"未接裁判"，得说清楚是哪一种。"""
    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"},
                   judge_fn=lambda q, a, s, r: {})
    assert rep.ragas_count == 1 and rep.ragas is None
    assert "裁判没给出可解析的分" in "\n".join(rep.to_lines())


def test_negative_samples_never_reach_the_judge():
    """负样本只判拒答：送进 RAGAS 会把四项均值无端拖低，分母也会对不上。"""
    seen = []

    def judge(question, answer, sources, reference):
        seen.append(question)
        return {"faithfulness": 1.0}

    golden = [{"question": "Q1", "expect": "甲"}, {"question": "N1", "negative": True}]
    table = {"Q1": {"answer": "甲"}, "N1": {"answer": "这个我不知道"}}
    rep = run_eval(golden, _answer_fn(table), judge_fn=judge)

    assert seen == ["Q1"]                                   # 负样本没送去裁判
    assert rep.ragas_count == 1
    assert "计入 1/1 条" in "\n".join(rep.to_lines())        # 分母只算正样本


def test_judge_gets_the_golden_reference_when_provided():
    got = {}

    def judge(question, answer, sources, reference):
        got["ref"] = reference
        return {}

    run_eval([{"question": "Q", "expect": "关键词", "reference": "完整的参考答案。"}],
             lambda q: {"answer": "甲"}, judge_fn=judge)
    assert got["ref"] == "完整的参考答案。"


def test_reference_falls_back_to_expect():
    got = {}

    def judge(question, answer, sources, reference):
        got["ref"] = reference
        return {}

    run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"}, judge_fn=judge)
    assert got["ref"] == "甲"


def test_report_states_the_context_recall_baseline():
    rep = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"},
                   judge_fn=lambda q, a, s, r: {"context_recall": 1.0})
    text = "\n".join(rep.to_lines())
    assert "reference" in text and "退化成 0/1" in text


# ---------- 报告口径不许撒谎（#52）----------

def test_the_embedding_label_reads_config_not_the_environment(monkeypatch):
    """`.env` 里的值**不会**进 os.environ，照环境变量渲染会把自己写成 fake。

    真机上出现过：跑的是 bge-m3，检索报告却写「嵌入: fake」—— 报告在口径上撒谎。
    """
    from app.config import get_settings
    from app.eval_core import embedding_label

    monkeypatch.setenv("EMBEDDING_PROVIDER", "环境变量里的假值")     # 环境变量说了不算
    monkeypatch.setattr(get_settings(), "embedding_provider", "siliconflow")
    monkeypatch.setattr(get_settings(), "embedding_model", "BAAI/bge-m3")

    assert embedding_label() == "siliconflow / BAAI/bge-m3"


def test_the_embedding_label_names_the_model_not_just_the_provider(monkeypatch):
    """只写 provider 不够 —— 换模型就换了口径，报告得看出**是哪个模型**。"""
    from app.config import get_settings
    from app.eval_core import embedding_label

    monkeypatch.setattr(get_settings(), "embedding_provider", "siliconflow")
    monkeypatch.setattr(get_settings(), "embedding_model", "BAAI/bge-m3")

    assert "bge-m3" in embedding_label()


def test_the_fake_provider_is_labelled_as_not_a_real_model(monkeypatch):
    """演示档别写成 bge-xxx —— 那看着像真跑了。"""
    from app.config import get_settings
    from app.eval_core import embedding_label

    monkeypatch.setattr(get_settings(), "embedding_provider", "fake")

    assert "不是真模型" in embedding_label()


def test_the_reports_do_not_render_the_provider_from_the_environment():
    """两处调用点都别再走 os.environ —— 那是这个 bug 的来源。

    用 AST 找**调用**而不是字符串匹配：后者换个引号或换行就漏（`tests/test_tools.py` 有同样教训）。
    """
    import ast
    import inspect

    import evaluate_retrieval
    import evaluate_rgb

    for mod in (evaluate_retrieval, evaluate_rgb):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called not in ("get", "getenv"):
                continue
            literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
            assert "EMBEDDING_PROVIDER" not in literals, (
                "%s 还在按环境变量渲染 provider —— 要读配置（embedding_label）" % mod.__name__)


def test_the_token_caliber_carries_the_reason_not_a_generic_placeholder():
    """没有真实分词器时，报告要写**为什么** —— 只印「（未接真实分词器）」等于没说（#53）。

    真机上原因就记在 note 里（SSLError: huggingface.co ...），却一直没被渲染出来。
    """
    from app.eval_core import ItemResult, Report

    reason = "真实分词器 Qwen/Qwen2.5-7B-Instruct 不可用：SSLError: 证书校验失败"
    item = ItemResult(question="q", expect="", answer="", fact_hit=False, grounded=False,
                      expect_page=None, pages=[], page_hit=None, ctx_note=reason)
    r = Report(items=[item])

    text = "".join(r._token_lines())

    assert "SSLError" in text
    assert "（未接真实分词器）" not in text


def _item(**kw):
    base = dict(question="q", expect="", answer="", fact_hit=False, grounded=False,
                expect_page=None, pages=[], page_hit=None)
    base.update(kw)
    from app.eval_core import ItemResult

    return ItemResult(**base)


def test_a_compression_exemption_does_not_swallow_the_tokenizer_reason():
    """**真机踩到的**（#53）：第一条问题恰好是枚举/编号查询（豁免压缩），

    「没有真实分词器」与「本问豁免压缩」原本挤在同一个 note 字段里，
    而报告级的口径取的是**第一条** note —— 于是把「豁免」当成了「没接分词器」的原因。
    """
    from app.eval_core import Report

    reason = "真实分词器 Qwen/xxx 不可用：SSLError: 证书校验失败"
    r = Report(items=[
        _item(ctx_exempt_note="本问为枚举/编号查询，豁免压缩：没有降幅可报"),   # 真机里它在最前面
        _item(ctx_note=reason),
    ])

    text = "".join(r._token_lines())

    assert "SSLError" in text                 # 真正的原因带出来了
    assert "豁免压缩" not in text.split("口径")[1][:80]


def test_the_reason_says_so_when_every_question_was_exempt():
    """全部豁免时，「不适用」的原因和「没有多轮历史」不是一回事 —— 别混成一句。"""
    from app.eval_core import Report

    r = Report(items=[_item(ctx_tokenizer="Qwen/xxx", ctx_budget=3000,
                            ctx_exempt_note="本问为枚举/编号查询，豁免压缩")])

    assert "全部豁免" in r.reduction_missing_reason
