"""一页评测报告（票 08 / #15）：一条命令拼齐所有段落，缺前置条件就明说。

四段脚本的 main 全部 stub 掉 —— 不联网、不建索引，只验拼装与「不造假」。
"""
from __future__ import annotations

import evaluate
import evaluate_all
import evaluate_latency
import evaluate_retrieval
import evaluate_rgb
from app.eval_core import ItemResult, Report


def _item(fact: bool) -> ItemResult:
    return ItemResult(question="Q", expect="甲", answer="甲" if fact else "乙",
                      fact_hit=fact, grounded=fact, expect_page=None, pages=[], page_hit=None)


def _stub_sections(monkeypatch, tmp_path):
    """把三段独立脚本换成写死的小报告，只验拼装。

    报告由**这段自己的 `main()` 写**（真脚本就是这样）—— 一页报告开跑前会先清掉上一轮的
    旧报告（票 39 / #48），所以夹具不能靠预先放一个文件。
    """
    for mod, name in ((evaluate_retrieval, "retrieval"), (evaluate_rgb, "rgb"),
                      (evaluate_latency, "latency")):
        path = tmp_path / ("%s.log" % name)
        monkeypatch.setattr(mod, "REPORT", str(path))

        def main(p=path, n=name):
            p.write_text("子报告 %s 的数字与口径\n" % n, encoding="utf-8")

        monkeypatch.setattr(mod, "main", main)


def _run(monkeypatch, tmp_path, out_name="summary.log"):
    out = tmp_path / out_name
    monkeypatch.setattr(evaluate_all, "REPORT", str(out))
    evaluate_all.main()
    return out.read_text(encoding="utf-8")


# ---------- 配置快照 ----------

def test_config_snapshot_lists_models_and_switches():
    snap = "\n".join(evaluate_all._config_snapshot())
    for key in ("llm_provider", "embedding_provider", "reranker_provider",
                "context_compress", "ragas_judge_model", "semantic_cache"):
        assert key in snap


# ---------- 目标线（只标注，不卡流程） ----------

def test_targets_mark_both_hit_and_miss():
    text = "\n".join(evaluate_all._target_lines(Report(items=[_item(True)])))
    assert "[达标]" in text

    text = "\n".join(evaluate_all._target_lines(Report(items=[_item(False)])))
    assert "[未达标]" in text


def test_target_line_says_so_when_a_kind_of_item_is_absent():
    text = "\n".join(evaluate_all._target_lines(Report(items=[_item(True)])))
    assert "本次没有这类数字" in text          # 没有负样本时拒答率那一行


def test_targets_say_generation_was_not_run():
    text = "\n".join(evaluate_all._target_lines(None))
    assert "本次没跑生成层" in text
    assert "[达标]" not in text and "[未达标]" not in text      # 绝不凭空给结论


# ---------- 拼装 ----------

def test_one_page_contains_config_targets_and_every_section(tmp_path, monkeypatch):
    rep = Report(items=[_item(True)])
    monkeypatch.setattr(evaluate, "run_online",
                        lambda *a, **k: (rep, {"status": "indexed", "chunk_count": 3, "page_count": 2}))
    monkeypatch.setattr(evaluate, "GOLDEN", "g.json")
    monkeypatch.setattr(evaluate, "DOC", "d.pdf")
    _stub_sections(monkeypatch, tmp_path)

    text = _run(monkeypatch, tmp_path)

    assert "=== 配置快照 ===" in text
    assert "=== 目标线（只作参照，不卡发版）===" in text
    assert "=== 生成层指标 ===" in text and "[达标]" in text
    assert "上传: indexed chunks=3 页数=2" in text
    for name in ("检索层（离线）", "RGB 中文四能力（离线）", "延迟（并发）"):
        assert "=== %s ===" % name in text
    assert "子报告 rgb 的数字与口径" in text           # 子报告的口径原样带过来


def test_missing_service_is_reported_and_the_page_still_lands(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("连不上 localhost:8000")

    monkeypatch.setattr(evaluate, "run_online", boom)
    _stub_sections(monkeypatch, tmp_path)
    text = _run(monkeypatch, tmp_path)

    assert "未跑：ConnectionError" in text
    assert "本次没跑生成层" in text          # 目标线也不给结论
    assert "答案含期望事实" in text           # 段落标题还在，只是明说没跑
    assert "[达标]" not in text


def test_section_that_fails_does_not_kill_the_page(tmp_path, monkeypatch):
    rep = Report(items=[_item(True)])
    monkeypatch.setattr(evaluate, "run_online",
                        lambda *a, **k: (rep, {"status": "indexed", "chunk_count": 1, "page_count": 1}))
    monkeypatch.setattr(evaluate, "GOLDEN", "g.json")
    monkeypatch.setattr(evaluate, "DOC", "d.pdf")
    _stub_sections(monkeypatch, tmp_path)

    def boom():
        raise RuntimeError("RGB 数据缺失")

    monkeypatch.setattr(evaluate_rgb, "main", boom)
    text = _run(monkeypatch, tmp_path)

    assert "未跑：RuntimeError: RGB 数据缺失" in text


def test_skip_env_marks_sections_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_SKIP", "延迟")
    rep = Report(items=[_item(True)])
    monkeypatch.setattr(evaluate, "run_online",
                        lambda *a, **k: (rep, {"status": "indexed", "chunk_count": 1, "page_count": 1}))
    monkeypatch.setattr(evaluate, "GOLDEN", "g.json")
    monkeypatch.setattr(evaluate, "DOC", "d.pdf")
    _stub_sections(monkeypatch, tmp_path)

    text = _run(monkeypatch, tmp_path)
    assert "按 EVAL_SKIP 跳过" in text


# ---------- 审查发现的几处（回归） ----------

def test_config_snapshot_covers_the_comparability_critical_keys():
    """两次报告要能比，就得把「真假模式 / 向量库 / Redis」这类也照下来。"""
    snap = "\n".join(evaluate_all._config_snapshot())
    for key in ("use_real", "vector_store", "redis_url", "rrf_k", "min_relevance"):
        assert key in snap


def test_target_table_drives_the_value_so_new_targets_cannot_silently_reuse_others():
    """目标线取值绑在表里 —— 不再按显示名分支，加一条不会落到别人的数字上。"""
    assert all(len(t) == 4 and callable(t[3]) for t in evaluate_all.TARGETS)
    keys = [t[0] for t in evaluate_all.TARGETS]
    assert keys == ["fact", "grounded", "refusal", "faithfulness"]


def test_faithfulness_target_uses_the_ragas_section():
    rep = Report(items=[_item(True)], judge_label="stub")
    rep.items[0].judged = {"faithfulness": 0.5}
    text = "\n".join(evaluate_all._target_lines(rep))
    assert "RAGAS 忠实度" in text and "[未达标]" in text


def test_skip_by_section_key_works_for_rgb(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_SKIP", "rgb")          # 文档里承诺的 key 必须真的生效
    rep = Report(items=[_item(True)])
    monkeypatch.setattr(evaluate, "run_online",
                        lambda *a, **k: (rep, {"status": "indexed", "chunk_count": 1, "page_count": 1}))
    monkeypatch.setattr(evaluate, "GOLDEN", "g.json")
    monkeypatch.setattr(evaluate, "DOC", "d.pdf")
    _stub_sections(monkeypatch, tmp_path)

    text = _run(monkeypatch, tmp_path)
    assert "=== RGB 中文四能力（离线） ===" in text
    assert "按 EVAL_SKIP 跳过" in text
    assert "子报告 rgb" not in text                  # 真跳过了，不是只加一行说明


def test_citation_coverage_shows_when_the_qa_implementation_provides_it():
    """spec 把引用覆盖率列为自研判据之一 —— 拿到了就要出现在页面上。"""
    item = _item(True)
    item.citation_coverage = 0.75
    text = "\n".join(Report(items=[item]).to_lines())
    assert "引用覆盖率" in text and "75%" in text


def test_citation_coverage_is_absent_when_not_available():
    text = "\n".join(Report(items=[_item(True)]).to_lines())
    assert "引用覆盖率(论断被来源支撑)" not in text     # 拿不到就不出这一行，不拿 0 顶替


# ---------- 压缩降幅（票 19）----------

def test_the_compression_reduction_sits_next_to_the_fact_row():
    """降幅紧挨事实命中 —— 只报降幅不报质量，等于奖励「把上下文砍掉」。"""
    item = _item(True)
    item.ctx_before, item.ctx_after, item.ctx_tokenizer = 100, 40, "Qwen/x"

    lines = evaluate_all._target_lines(Report(items=[item]))
    i = next(k for k, ln in enumerate(lines) if "答案含期望事实" in ln and "实际" in ln)

    assert lines[i + 1].startswith("  上下文压缩降幅(token)")
    assert "60%" in lines[i + 1] and "Qwen/x" in lines[i + 1]      # 口径跟着数字


def test_the_reduction_row_says_unavailable_without_a_real_tokenizer():
    row = next(ln for ln in evaluate_all._target_lines(Report(items=[_item(True)]))
               if "上下文压缩降幅" in ln)

    assert "不可用" in row and "%" not in row


def test_a_working_tokenizer_with_nothing_to_compress_is_not_called_missing():
    """有分词器、只是这轮没压到东西 —— 不许写成「没有真实分词器」（原因取自评测核心那一处）。"""
    item = _item(True)
    item.ctx_before, item.ctx_after, item.ctx_tokenizer = 0, 0, "Qwen/x"

    row = next(ln for ln in evaluate_all._target_lines(Report(items=[item]))
               if "上下文压缩降幅" in ln)

    assert "不适用" in row and "Qwen/x" in row
    assert "没有真实分词器" not in row


def test_a_stale_report_from_a_previous_run_is_not_left_behind(tmp_path, monkeypatch):
    """某段跑不动时，摘要会写「未跑」—— 但磁盘上**上一轮**的报告也必须清掉。

    否则打开那个文件看到的是一批正常数字，读者根本不会知道这一段这次没跑成（票 39 / #48）。
    """
    stale = tmp_path / "rgb.log"
    stale.write_text("上一轮的正常数字\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_rgb, "REPORT", str(stale))

    def boom():
        raise RuntimeError("数据没在本地")

    monkeypatch.setattr(evaluate_rgb, "main", boom)

    out = evaluate_all._section("RGB 中文四能力（离线）", evaluate_rgb, str(stale))

    assert any("未跑" in line for line in out)
    assert not stale.exists()          # 旧的那份不许留着骗人
