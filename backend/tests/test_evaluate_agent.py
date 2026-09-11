"""对比 runner（票 15 / #22）：本地两条链路都跑得通；缺输入就如实说未跑。"""
from __future__ import annotations

import json

import evaluate_agent

DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。"


def _inputs(tmp_path, expect: str = "803.96"):
    doc = tmp_path / "annual.txt"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "营收多少？", "expect": expect, "page": 1}],
                                 ensure_ascii=False), encoding="utf-8")
    return str(golden), str(doc), str(tmp_path / "report.log")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_missing_inputs_are_reported_as_not_run(tmp_path):
    """缺前置就写「未跑」—— 绝不凭空造数字。"""
    _, _, report = _inputs(tmp_path)
    evaluate_agent.main(golden=str(tmp_path / "nope.json"), doc=str(tmp_path / "nope.pdf"),
                        report=report)
    text = _read(report)

    assert "未跑" in text
    assert "事实命中率" not in text and "%" not in text


def test_both_links_run_offline_on_the_same_golden_set(tmp_path):
    """两条链路真的各跑了一遍：对比表里有数字、有延迟，且写明是演示模式。"""
    golden, doc, report = _inputs(tmp_path, expect="模拟回答")
    evaluate_agent.main(golden=golden, doc=doc, report=report)
    text = _read(report)

    assert "确定性链路" in text and "代理链路" in text
    assert "事实命中率" in text and "100%" in text          # 假模型的固定答案里确实有这串字
    assert "平均总耗时" in text and "ms" in text
    assert "演示模式" in text                               # 口径写清楚，数字才不会被误读
    assert "黄金集 1 条：正样本 1 / 负样本 0" in text
