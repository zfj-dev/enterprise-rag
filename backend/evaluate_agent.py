"""代理链路 vs 确定性链路：同一黄金集、同一个评测核心，出「事实命中 / 拒答 / 延迟」对比。

**不依赖运行中的服务**：本地起运行时、把文档真走一遍入库管线，两条链路各跑一遍黄金集。
指标与排版在 app/eval_compare.py（数字全部来自评测核心 app/eval_core.py）。

用法:  cd backend && python evaluate_agent.py        （或 scripts/evaluate_agent.ps1）
环境:  EVAL_GOLDEN / EVAL_DOC 同 evaluate.py；报告路径 EVAL_AGENT_REPORT
报告:  logs/agent-vs-baseline.log（助手可读）
"""
from __future__ import annotations

import json
import os

from app.eval_agent import agent_answer_fn, deterministic_answer_fn
from app.eval_compare import compare_links
from app.eval_setup import drop_kb, ensure_schema, eval_user, ingest_file, new_kb

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.environ.get("EVAL_AGENT_REPORT",
                         os.path.join(BACKEND, "logs", "agent-vs-baseline.log"))
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
USERNAME = "__agent_eval__"


def _config_lines() -> list[str]:
    from app.config import get_settings

    s = get_settings()
    return [
        "模型: LLM_PROVIDER=%s model=%s" % (s.llm_provider, s.llm_model),
        "嵌入=%s 重排=%s" % (s.embedding_provider, s.reranker_provider),
        "代理开关 AGENT_ENABLED=%s，代理步数上限=%d" % (s.agent_enabled, s.agent_max_steps),
    ]


def _write(report: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(report)


def main(golden: str | None = None, doc: str | None = None, report: str | None = None) -> None:
    golden = golden or GOLDEN
    doc = doc or DOC
    report = report or REPORT
    lines = ["=== 代理链路 vs 确定性链路（同一黄金集）===",
             "黄金集: %s" % golden,
             "文档: %s" % doc,
             "报告: %s" % report]
    lines += _config_lines()
    lines.append("")

    if not os.path.exists(golden) or not os.path.exists(doc):
        lines += ["未跑：缺输入。",
                  "  黄金集 %s：%s" % (golden, "有" if os.path.exists(golden) else "**缺**"),
                  "  文档 %s：%s" % (doc, "有" if os.path.exists(doc) else "**缺**"),
                  "补上前置再跑 —— 这里不会拿假数据顶替。"]
        _write(report, lines)
        return

    with open(golden, encoding="utf-8") as f:
        goldenset = json.load(f)

    from app.config import get_settings
    from app.core.container import build_runtime
    from app.db.session import SessionLocal

    ensure_schema()
    rt = build_runtime()
    db = SessionLocal()
    kb = None
    try:
        user = eval_user(db, USERNAME)
        kb = new_kb(db, user.id, "代理对比评测库")
        doc_id = ingest_file(db, rt, doc, user.id, kb.id)
        lines.append("文档已入库：doc_id=%s" % doc_id)
        if get_settings().llm_provider == "fake":
            lines.append("注意：LLM_PROVIDER=fake（演示模式）—— 两条链路的答案都是固定文本，"
                         "这些数字只证明链路跑得通，不代表回答质量。")
        lines.append("")

        links = {
            "确定性链路": deterministic_answer_fn(db, rt, user, kb.id),
            "代理链路": agent_answer_fn(db, rt, user, kb.id,
                                        max_steps=get_settings().agent_max_steps),
        }
        lines.extend(compare_links(
            goldenset, links,
            note="注意：代理列只计**代理循环本身**；确定性列计完整单步管线（改写 / 检索 / 重排 / 生成）"
                 "—— 这一列对代理偏乐观。"))
    finally:
        if kb is not None:
            try:
                drop_kb(db, kb.id)
            except Exception as e:      # noqa: BLE001 —— 清理失败不该毁掉已算出的报告
                print("清库失败（不影响报告）：%s" % e)
        db.close()
    _write(report, lines)


if __name__ == "__main__":
    main()
