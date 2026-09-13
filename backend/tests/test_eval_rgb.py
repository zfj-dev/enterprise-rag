"""RGB 中文四能力适配器（票 06 / #13）：格式转换 + 数据缺失时明说。

用官方 schema 的小样本做 fixture —— 不联网、不改官方数据。
"""
from __future__ import annotations

import json

NL = chr(10)   # 只在这里拼换行，避免转义层数出错

import pytest

from app.eval_core import run_eval
from app.eval_rgb import (DatasetMissing, available, dataset_path,
                          load_entries)

# 夹具**照官方文件的真实形状**：answer 是**列表**、positive 可以是**list 套 list**。
# 早期版本把 answer 写成字符串、文档写成单层列表，测试就一路绿着 ——
# 真机一跑 `Extra data`（官方是 JSONL）+ 命中率全 0（str(list) 匹配不上），见 #52。
ZH = [
    {"id": 0, "query": "香港第六届立法会选举有多少个议席？", "answer": ["70"],
     "positive": ["第六届立法会共有70个议席，经2016年9月4日的选举产生。"],
     "negative": ["今天的天气很好。"]},
    {"id": 1, "query": "第二个问题", "answer": ["甲"], "positive": ["甲"], "negative": []},
]
INT = [{"id": 0, "query": "综合两篇文档回答", "answer": ["乙"],
        "positive": [["乙的第一半", "乙的第二半"]], "negative": ["噪声"]}]
FACT = [{"id": 0, "query": "议席有多少", "answer": "70", "fakeanswer": "170",
         "positive": ["共有70个议席"], "positive_wrong": ["共有170个议席", "错得离谱的另一篇"]}]


def _dir(tmp_path, **files):
    """写夹具 —— **一行一条（JSONL）**，跟官方文件一致。

    夹具要是照着「我以为的格式」写，测试就只验证「我以为的格式」：
    官方其实是 JSONL，整份当单个 JSON 读会直接报 `Extra data`（#52）。
    """
    for name, data in files.items():
        body = "".join(json.dumps(item, ensure_ascii=False) + NL for item in data)
        (tmp_path / name).write_text(body, encoding="utf-8")
    return str(tmp_path)


def _full(tmp_path):
    return _dir(tmp_path, **{"zh.json": ZH, "zh_int.json": INT, "zh_fact.json": FACT})


# ---------- 格式转换 ----------

def test_noise_entries_are_converted_with_their_docs(tmp_path):
    entries = load_entries(_full(tmp_path), "noise")

    assert len(entries) == 2
    first = entries[0]
    assert first["question"] == ZH[0]["query"]
    assert first["expect"] == "70"
    assert first["group"] == "噪声鲁棒"
    assert "negative" not in first
    assert first["documents"] == ZH[0]["positive"] + ZH[0]["negative"]


def test_rejection_ability_only_gets_noise_docs_and_expects_refusal(tmp_path):
    entries = load_entries(_full(tmp_path), "rejection")

    first = entries[0]
    assert first["negative"] is True
    assert "expect" not in first                       # 文档里根本没有答案，不设期望事实
    assert first["documents"] == ZH[0]["negative"]      # 官方拒绝协议：只给噪声文档
    assert first["group"] == "否定拒绝"


def test_integration_entries_come_from_the_int_file(tmp_path):
    entries = load_entries(_full(tmp_path), "integration")
    assert entries[0]["question"] == "综合两篇文档回答"
    assert entries[0]["documents"] == ["乙的第一半", "乙的第二半", "噪声"]
    assert entries[0]["group"] == "信息集成"


def test_counterfactual_entries_keep_the_wrong_docs(tmp_path):
    """反事实鲁棒：与事实相悖的文档也要喂进去 —— 考验的是别被带偏。"""
    entries = load_entries(_full(tmp_path), "counterfactual")
    assert entries[0]["expect"] == "70"
    assert entries[0]["documents"] == ["共有70个议席", "共有170个议席", "错得离谱的另一篇"]
    assert entries[0]["group"] == "反事实鲁棒"


def test_adapter_does_not_touch_the_original_data(tmp_path):
    root = _full(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for ability in ("noise", "rejection", "integration", "counterfactual"):
        load_entries(root, ability)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert before == after


# ---------- 数据缺失：明说，不造假 ----------

def test_missing_dataset_raises_with_an_actionable_message(tmp_path):
    with pytest.raises(DatasetMissing) as e:
        load_entries(str(tmp_path), "integration")
    msg = str(e.value)
    assert "zh_int.json" in msg and "chen700564/RGB" in msg


def test_available_reports_which_abilities_can_run(tmp_path):
    root = _dir(tmp_path, **{"zh.json": ZH})
    got = available(root)
    assert got["noise"] is True and got["rejection"] is True      # 同一份 zh.json
    assert got["integration"] is False and got["counterfactual"] is False


def test_unknown_ability_is_a_programming_error(tmp_path):
    with pytest.raises(KeyError):
        dataset_path(str(tmp_path), "不存在的能力")


# ---------- 接进核心：四能力各自的数字 ----------

def _answer_fn(table):
    return lambda q: table[q]


def test_report_breaks_the_numbers_down_by_ability(tmp_path):
    entries = (load_entries(_full(tmp_path), "noise")[:1]
               + load_entries(_full(tmp_path), "rejection")[:1])
    table = {ZH[0]["query"]: {"answer": "共有 70 个议席 [来源: a, 第1页]",
                             "sources": [{"text": "共有70个议席", "page": 1}]}}
    rep = run_eval(entries, _answer_fn(table))

    assert rep.group_names == ["噪声鲁棒", "否定拒绝"]
    text = "\n".join(rep.to_lines())
    assert "=== 分能力 ===" in text
    assert "噪声鲁棒" in text and "否定拒绝" in text
    assert "100%" in text            # 噪声鲁棒命中；否定拒绝那条空答不算拒答


def test_rejection_ability_reuses_the_refusal_judge(tmp_path):
    entries = load_entries(_full(tmp_path), "rejection")[:1]
    table = {ZH[0]["query"]: {"answer": "根据现有资料无法确定（未检索到可引用的内容）。",
                              "sources": []}}
    rep = run_eval(entries, _answer_fn(table))

    assert rep.negatives[0].refused is True          # 复用票 03 的拒答判据
    assert rep.sub("否定拒绝").refuse_rate == 1.0
    assert "100%" in "\n".join(rep.to_lines())


# ---------- 运行脚本里那个「按顺序取库」的缝 ----------

def test_corrupted_json_is_reported_as_dataset_missing(tmp_path):
    (tmp_path / "zh_int.json").write_text("{坏掉的", encoding="utf-8")
    with pytest.raises(DatasetMissing) as e:
        load_entries(str(tmp_path), "integration")
    assert "zh_int.json" in str(e.value)


def test_noise_and_rejection_share_questions_so_never_key_a_kb_by_question(tmp_path):
    """RGB 的坑：噪声鲁棒与否定拒绝是**同一批问题** —— 按问题名建 KB 映射必然串台。"""
    root = _full(tmp_path)
    noise = load_entries(root, "noise")
    rejection = load_entries(root, "rejection")

    assert [e["question"] for e in noise] == [e["question"] for e in rejection]
    assert noise[0]["documents"] != rejection[0]["documents"]      # 但喂进去的文档不一样


def test_answer_cursor_gives_each_entry_its_own_kb():
    from evaluate_rgb import _answer_cursor

    entries = [{"question": "同一个问题"}, {"question": "同一个问题"}]
    seen = []

    def answer_with(kb_id, question):
        seen.append(kb_id)
        return {"answer": "", "sources": []}

    ask = _answer_cursor(entries, ["kb_noise", "kb_reject"], answer_with)
    ask("同一个问题")
    ask("同一个问题")

    assert seen == ["kb_noise", "kb_reject"]       # 各取各的库，没串台


def test_answer_cursor_refuses_to_answer_out_of_order():
    from evaluate_rgb import _answer_cursor

    ask = _answer_cursor([{"question": "甲"}], ["kb"], lambda kb, q: {"answer": ""})
    with pytest.raises(RuntimeError):
        ask("乙")                                   # 顺序对不上就炸，绝不静默答错


# ---------- 官方数据是 JSONL（#52）----------

def test_official_jsonl_files_are_read_line_by_line(tmp_path):
    """官方文件**一行一条**，不是单个 JSON 文档 —— 夹具现在就是这个形状。"""
    assert len(load_entries(_full(tmp_path), "noise")) == len(ZH)


def test_a_single_json_array_still_works(tmp_path):
    """有的数据版本/别处导出的确是单个数组 —— 两种都认，别为兼容把老格式弄坏。"""
    for name, data in (("zh.json", ZH), ("zh_int.json", INT), ("zh_fact.json", FACT)):
        (tmp_path / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert len(load_entries(str(tmp_path), "noise")) == len(ZH)


def test_a_broken_line_names_the_line_number(tmp_path):
    """坏行要报出**行号** —— 几 MB 的文件里没有行号没法找。"""
    body = json.dumps(INT[0], ensure_ascii=False) + NL + "{不是 JSON" + NL
    (tmp_path / "zh_int.json").write_text(body, encoding="utf-8")

    with pytest.raises(DatasetMissing) as e:
        load_entries(str(tmp_path), "integration")
    assert "第 2 行" in str(e.value)


def test_a_list_answer_does_not_become_a_python_repr(tmp_path):
    """**最要命的一条**：`str(["12名"])` 得到字面量 `['12名']`，跟答案文本永远匹配不上。

    而报告只会显示「命中率 0」—— 看上去像效果差，看不出是解析坏了。
    """
    item = {"id": 0, "query": "问", "answer": ["12名"],
            "positive": ["答案就是12名"], "negative": []}
    (tmp_path / "zh_int.json").write_text(json.dumps(item, ensure_ascii=False) + NL,
                                          encoding="utf-8")

    expect = load_entries(str(tmp_path), "integration")[0]["expect"]

    assert expect == "12名"
    assert "[" not in expect and "'" not in expect


def test_nested_document_lists_are_flattened(tmp_path):
    """`zh_int.json` 的 positive 是 **list 套 list** —— 摊平，别把子列表 str 进去。"""
    item = {"id": 0, "query": "问", "answer": ["乙"],
            "positive": [["甲的第一半", "甲的第二半"], ["乙全文"]], "negative": []}
    (tmp_path / "zh_int.json").write_text(json.dumps(item, ensure_ascii=False) + NL,
                                          encoding="utf-8")

    docs = load_entries(str(tmp_path), "integration")[0]["documents"]

    assert docs == ["甲的第一半", "甲的第二半", "乙全文"]

