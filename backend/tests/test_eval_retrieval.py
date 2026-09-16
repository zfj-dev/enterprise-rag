"""检索层指标（票 04 / #11）：hit@k / recall@k / MRR 与负样本 top-1。

检索方式与打分全部注入 stub —— 不碰服务、不碰真实嵌入，结果确定。
"""
from __future__ import annotations

import json

import pytest

from app.eval_core import (Report, hit_recall_at_k, reciprocal_rank,
                           run_eval, run_retrieval_eval)

KS = (1, 3)
LABELS4 = ("hybrid+rerank", "hybrid", "vector", "bm25")


def _fn(table):
    return lambda question: table[question]


# ---------- 单问题口径 ----------

def test_hit_recall_at_k_units():
    ranked = ["a", "b", "c"]
    assert hit_recall_at_k(ranked, {"b"}, 1) == (0.0, 0.0)       # b 不在前 1
    assert hit_recall_at_k(ranked, {"b"}, 3) == (1.0, 1.0)
    assert hit_recall_at_k(ranked, {"a", "c"}, 2) == (1.0, 0.5)   # 命中 1 个 / 共 2 个
    assert hit_recall_at_k(ranked, set(), 3) == (0.0, 0.0)


def test_reciprocal_rank_uses_the_whole_ranking():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, {"a"}) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


# ---------- 多方式 × 多 k ----------

def test_all_four_retrieval_labels_are_kept():
    ranks = {label: ["g1", "x"] for label in LABELS4}
    m = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}],
                           _fn({"Q1": {"ranks": ranks, "top1": 0.9}}), ks=KS)
    assert m.labels == LABELS4
    assert all(label in m.scores for label in LABELS4)


def test_a_gold_set_bigger_than_k_says_recall_at_k_is_capped():
    """正确分块比 k 还多的题：recall@k 的上限就是 k/正确分块数，结构性地低。

    报告必须点名说清 —— 否则读者会把这几个数当成「整体检索水平」（#60）。
    （真机上「论文的研究对象」正确分块 102 条，recall@10 上限只有 0.10。）
    """
    ranks = {label: ["g1", "g2"] for label in LABELS4}
    m = run_retrieval_eval(
        [{"question": "有哪些图片", "gold_ids": {"g%d" % i for i in range(1, 51)}}],
        _fn({"有哪些图片": {"ranks": ranks, "top1": 0.9}}), ks=(1, 3))
    text = "\n".join(m.to_lines())

    assert "上限" in text and "有哪些图片" in text
    assert "0.06" in text                     # 3 / 50
    assert "hit@k" in text                    # 说清哪几项不受影响


def test_a_gold_set_within_k_gets_no_cap_warning():
    """装得下的题不许出现这条提示 —— 提示一多就成了噪音。"""
    ranks = {label: ["g1", "x"] for label in LABELS4}
    m = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}],
                           _fn({"Q1": {"ranks": ranks, "top1": 0.9}}), ks=KS)

    assert "注意：" not in "\n".join(m.to_lines())


def test_metrics_are_averaged_over_questions_per_label_and_k():
    table = {
        "Q1": {"ranks": {"hybrid": ["g1", "x"], "vector": ["x", "g1"]}, "top1": 0.9},
        "Q2": {"ranks": {"hybrid": ["x", "y"], "vector": ["x", "y"]}, "top1": 0.2},
    }
    questions = [{"question": "Q1", "gold_ids": {"g1"}}, {"question": "Q2", "gold_ids": {"g2"}}]
    m = run_retrieval_eval(questions, _fn(table), ks=KS)

    assert m.scored_count == 2 and m.skipped_count == 0
    assert m.labels == ("hybrid", "vector")
    assert m.scores["hybrid"].hit[1] == pytest.approx(0.5)      # Q1 命中、Q2 没命中
    assert m.scores["hybrid"].hit[3] == pytest.approx(0.5)
    assert m.scores["vector"].hit[1] == pytest.approx(0.0)      # Q1 的 g1 排第 2
    assert m.scores["vector"].hit[3] == pytest.approx(0.5)
    assert m.scores["hybrid"].mrr == pytest.approx(0.5)         # 1.0 与 0.0
    assert m.scores["vector"].mrr == pytest.approx(0.25)        # 0.5 与 0.0


def test_k_is_configurable():
    table = {"Q1": {"ranks": {"hybrid": ["x", "g1"]}, "top1": 0.9}}
    m = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}], _fn(table), ks=(2,))
    assert m.ks == (2,)
    assert list(m.scores["hybrid"].hit) == [2]
    assert m.scores["hybrid"].hit[2] == 1.0
    assert "hit@2" in "\n".join(m.to_lines())


def test_empty_ranking_scores_zero_without_crashing():
    table = {"Q1": {"ranks": {"hybrid": []}, "top1": None}}
    m = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}], _fn(table), ks=KS)
    assert m.scores["hybrid"].hit[1] == 0.0
    assert m.scores["hybrid"].mrr == 0.0
    assert "[MISS]" in "\n".join(m.to_lines())


def test_labels_are_discovered_across_questions():
    table = {"Q1": {"ranks": {"hybrid": ["g1"]}, "top1": 0.9},
             "Q2": {"ranks": {"hybrid": ["g2"], "bm25": ["g2"]}, "top1": 0.9}}
    questions = [{"question": "Q1", "gold_ids": {"g1"}}, {"question": "Q2", "gold_ids": {"g2"}}]
    m = run_retrieval_eval(questions, _fn(table), ks=KS)
    assert m.labels == ("hybrid", "bm25")


# ---------- 判不出正确分块：不静默 ----------

def test_skipped_questions_are_named_not_just_counted():
    table = {"Q1": {"ranks": {"hybrid": ["g1"]}, "top1": 0.5},
             "Q2": {"ranks": {"hybrid": ["x"]}, "top1": 0.5}}
    questions = [{"question": "Q1", "gold_ids": {"g1"}}, {"question": "Q2", "gold_ids": set()}]
    m = run_retrieval_eval(questions, _fn(table), ks=KS)

    assert m.scored_count == 1 and m.skipped_count == 1
    assert m.skipped_questions == ["Q2"]
    text = "\n".join(m.to_lines())
    assert "1 条因判不出正确分块未计入" in text and "Q2" in text


# ---------- 负样本：向量 top-1 与参照阈值 ----------

def test_negative_samples_are_checked_against_the_threshold():
    table = {"N1": {"ranks": {"vector": ["x"]}, "top1": 0.11, "top1_id": "c_x"},
             "N2": {"ranks": {"vector": ["y"]}, "top1": 0.62, "top1_id": "c_y"}}
    questions = [{"question": "N1", "negative": True}, {"question": "N2", "negative": True}]
    m = run_retrieval_eval(questions, _fn(table), ks=KS, threshold=0.4)

    assert m.scored_count == 0                                  # 负样本不进检索指标的分母
    assert [n.below for n in m.negatives] == [True, False]
    assert [n.chunk_id for n in m.negatives] == ["c_x", "c_y"]
    text = "\n".join(m.to_lines())
    assert "低(该拒)" in text and "高(会被当成相关内容)" in text
    assert "c_x" in text                                        # 哪个块被捞上来也要看得到
    assert "不是系统真正的拒答条件" in text                       # 阈值名不副实，报告里要说明白


def test_without_threshold_only_the_similarity_is_recorded():
    table = {"N1": {"ranks": {"vector": ["x"]}, "top1": 0.11}}
    m = run_retrieval_eval([{"question": "N1", "negative": True}], _fn(table), ks=KS)

    assert all(n.below is None for n in m.negatives)
    assert "未给参照阈值" in "\n".join(m.to_lines())


def test_no_negative_samples_says_so():
    m = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}],
                           _fn({"Q1": {"ranks": {"hybrid": ["g1"]}, "top1": 0.9}}), ks=KS)
    assert "本次没有负样本条目" in "\n".join(m.to_lines())


# ---------- 逐条明细：能定位「漏了哪题、期望是什么」 ----------

def test_rows_carry_expect_and_doc_for_locating_misses():
    table = {"Q1": {"ranks": {"hybrid": ["x"]}, "top1": 0.3}}
    m = run_retrieval_eval([{"question": "Q1", "doc": "a.pdf", "expect": "RTX 3060",
                             "gold_ids": {"g1"}}], _fn(table), ks=KS)

    row = m.rows[0]
    assert (row.question, row.doc, row.expect, row.gold_count) == ("Q1", "a.pdf", "RTX 3060", 1)
    text = "\n".join(m.to_lines())
    assert "期望:RTX 3060" in text and "a.pdf" in text


# ---------- 生成层与检索层在同一份报告里 ----------

def test_one_report_can_hold_both_generation_and_retrieval():
    gen = run_eval([{"question": "Q1", "expect": "甲"}],
                   lambda q: {"answer": "甲", "sources": []})
    ret = run_retrieval_eval([{"question": "Q1", "gold_ids": {"g1"}}],
                             _fn({"Q1": {"ranks": {"hybrid": ["g1"]}, "top1": 0.9}}), ks=KS)

    text = "\n".join(Report(items=gen.items, retrieval=ret).to_lines())
    assert "=== 生成层指标 ===" in text
    assert "=== 检索层指标 ===" in text
    assert "答案含期望事实" in text
    assert text.index("=== 生成层指标 ===") < text.index("=== 检索层指标 ===")


def test_report_without_retrieval_keeps_its_old_shape():
    gen = run_eval([{"question": "Q1", "expect": "甲"}], lambda q: {"answer": "甲", "sources": []})
    text = "\n".join(gen.to_lines())
    assert "=== 生成层指标 ===" not in text      # 没有检索段就不加抬头，既有输出不变
    assert "=== 检索层指标 ===" not in text


# ---------- 脚本侧的 gold 判定（口径复用核心的 normalize） ----------

def test_gold_ids_match_on_content_and_page():
    from evaluate_retrieval import _gold_ids

    chunks = [{"id": "a", "content": "GPU：RTX 3060", "page_num": 29},
              {"id": "b", "content": "RTX 3060", "page_num": 31},
              {"id": "c", "content": "无关", "page_num": 29}]

    assert _gold_ids(chunks, {"expect": "RTX 3060"}) == {"a", "b"}
    assert _gold_ids(chunks, {"expect": "RTX  3060", "page": 29}) == {"a"}   # 空格容错 + 页码过滤
    assert _gold_ids(chunks, {"expect": "没有这个"}) == set()


def test_two_documents_do_not_share_chunk_ids(tmp_path):
    """多文档评测：每份文档各自的 doc_id。

    都写死 `evaldoc` 时，块 id（`{doc_id}_p{页}_{序}`）在两份文档间完全重合，
    进内存向量库是**按 id 覆盖**（`_data[id] = ...`）—— 第二份文档直接把第一份盖掉，
    报告里的检索数字于是只反映最后一份。用 EVAL_DOCS 跑多文档时这条必中（#64 批 4）。
    """
    from app.core.container import build_runtime
    from evaluate_retrieval import KB_ID, OWNER_ID, _index_doc

    a = tmp_path / "甲.txt"
    a.write_text("甲公司的营业收入为100亿元。", encoding="utf-8")
    b = tmp_path / "乙.txt"
    b.write_text("乙公司的营业收入为200亿元。", encoding="utf-8")

    rt = build_runtime()
    ca = _index_doc(rt, str(a), "evaldoc_1")
    cb = _index_doc(rt, str(b), "evaldoc_2")

    assert ca and cb
    assert {c["id"] for c in ca}.isdisjoint({c["id"] for c in cb})
    hit = rt.vector_store.search(rt.embedding.encode(["营业收入"])[0], top_k=50,
                                 filter_meta={"kb_id": KB_ID, "owner_id": OWNER_ID})
    assert len(hit) == len(ca) + len(cb)          # 两份都在库里，谁也没盖掉谁

    bm = rt.bm25.search("营业收入", top_k=50,                 # BM25 也是按 id 存的，同样要查
                        filter_meta={"kb_id": KB_ID, "owner_id": OWNER_ID})
    assert {c["id"] for c in ca} | {c["id"] for c in cb} <= {d["chunk_id"] for d in bm}


def test_the_multi_doc_run_derives_a_distinct_doc_id_per_document(tmp_path, monkeypatch):
    """跑真路径（`EVAL_DOCS` 逗号分隔）—— 只测 `_index_doc` 的入参，改回写死也照样绿。"""
    import re

    import evaluate_retrieval as ev

    a = tmp_path / "甲.txt"
    a.write_text("甲公司的营业收入为100亿元。", encoding="utf-8")
    b = tmp_path / "乙.txt"
    b.write_text("乙公司的营业收入为200亿元。", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "营收多少", "expect": "100亿元", "doc": "甲.txt"}],
                                 ensure_ascii=False), encoding="utf-8")
    report = tmp_path / "retrieval.log"

    monkeypatch.setenv("EVAL_DOCS", "%s,%s" % (a, b))
    monkeypatch.setattr(ev, "GOLDEN", str(golden))
    monkeypatch.setattr(ev, "REPORT", str(report))
    ev._main_body()

    text = report.read_text(encoding="utf-8")
    assert sorted(re.findall(r"doc_id=([^)\s]+)", text)) == ["evaldoc_1", "evaldoc_2"]


def test_duplicate_filenames_are_reported_as_not_run(tmp_path, monkeypatch):
    """两份 EVAL_DOCS 重名：黄金集按文件名认领文档，认不出就写「未跑」—— 不许猜一份接着算。"""
    import evaluate_retrieval as ev

    d1, d2 = tmp_path / "甲目录", tmp_path / "乙目录"
    d1.mkdir()
    d2.mkdir()
    (d1 / "报告.txt").write_text("甲的营业收入为100亿元。", encoding="utf-8")
    (d2 / "报告.txt").write_text("乙的营业收入为200亿元。", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "营收多少", "expect": "100亿元"}],
                                 ensure_ascii=False), encoding="utf-8")
    report = tmp_path / "retrieval.log"

    monkeypatch.setenv("EVAL_DOCS", "%s,%s" % (d1 / "报告.txt", d2 / "报告.txt"))
    monkeypatch.setattr(ev, "GOLDEN", str(golden))
    monkeypatch.setattr(ev, "REPORT", str(report))
    ev._main_body()

    text = report.read_text(encoding="utf-8")
    assert "未跑" in text and "重名" in text and "报告.txt" in text
    assert "hit@" not in text           # 一个指标都没算出来，不许拿错的那份顶数
