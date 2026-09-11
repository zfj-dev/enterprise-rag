"""质量护栏（票 22）：同一黄金集跑**压缩前 / 压缩后**，事实命中与降幅并列，下降即判未通过。

**不依赖运行中的服务**：本地起运行时、把文档真走一遍入库管线，两条配置各跑一遍黄金集。
两条**配置**用的是同一条链路（本地确定性管线），只有压缩开关不同 —— 差异归因才干净。
指标与排版取自评测核心（app/eval_core.py）与对比模块（app/eval_compare.py）。

跑法固定**多轮**（同一会话连着问）：单轮每题新开会话，压根没有可压的历史，
那份「护栏」就是空的 —— 压缩要有历史可压才谈得上。

用法:  cd backend && python evaluate_guardrail.py   （或 scripts/evaluate_guardrail.ps1）
环境:  EVAL_GOLDEN / EVAL_DOC 同 evaluate.py；报告路径 EVAL_GUARDRAIL_REPORT
报告:  logs/compression-guardrail.log（助手可读）
"""
from __future__ import annotations

import json
import os

from app.eval_agent import deterministic_answer_fn
from app.eval_compare import guardrail_lines, render_compare, run_links
from app.eval_setup import drop_kb, ensure_schema, eval_user, ingest_file, new_kb

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.environ.get("EVAL_GUARDRAIL_REPORT",
                        os.path.join(BACKEND, "logs", "compression-guardrail.log"))
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
USERNAME = "__guardrail_eval__"


def _with_compress(ask, on: bool):
    """把这次问答的压缩开关**临时**扳到 on/off —— 两条配置共用同一套管线，只差这一个开关。

    `get_settings()` 是进程级单例，所以这两条链路**只能顺序跑**（`run_links` 正是顺序的）；
    若将来并发跑多条链路，列与开关会互相串台 —— 那时得把开关做成显式参数而非全局态。
    """
    from app.config import get_settings

    def wrapped(question: str) -> dict:
        s = get_settings()
        was = s.context_compress
        s.context_compress = on
        try:
            return ask(question)
        finally:
            s.context_compress = was

    return wrapped


def _config_lines() -> list[str]:
    from app.config import get_settings

    s = get_settings()
    return [
        "模型: LLM_PROVIDER=%s model=%s" % (s.llm_provider, s.llm_model),
        "上下文预算 %d tokens；保留最近 %d 轮" % (s.context_token_budget, s.context_keep_recent),
        "分词器（配置）: %s" % (s.tokenizer_model or "（未配置 —— 降幅那一行会写「不可用」）"),
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
    lines = ["=== 质量护栏：压缩前 vs 压缩后（同一黄金集）===",
             "黄金集: %s" % golden,
             "文档: %s" % doc,
             "报告: %s" % report,
             "跑法: 多轮（同一会话连着问，历史累积才压得起来 —— 单轮每题新开会话没有可压的历史）"]
    lines += _config_lines()

    if not os.path.exists(golden) or not os.path.exists(doc):
        lines += ["", "未跑：缺输入。",
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
    # 口径写「实际」而不是「配置」：模型配了但加载失败时，报告数字与配置名就对不上了
    lines.append("分词器（实际）: %s"
                 % (getattr(rt.token_counter, "label", "") or "无 —— 降幅那行会写「不可用」"))
    lines.append("")
    db = SessionLocal()
    kb = None
    try:
        user = eval_user(db, USERNAME)
        kb = new_kb(db, user.id, "压缩护栏评测库")
        doc_id = ingest_file(db, rt, doc, user.id, kb.id)
        lines.append("文档已入库：doc_id=%s" % doc_id)
        if get_settings().llm_provider == "fake":
            lines.append("注意：LLM_PROVIDER=fake（演示模式）—— 两条配置的答案都是固定文本，"
                         "这些数字只证明链路跑得通，不代表回答质量。")
        lines.append("")

        links = {
            "压缩前": _with_compress(
                deterministic_answer_fn(db, rt, user, kb.id, "guardrail-off"), False),
            "压缩后": _with_compress(
                deterministic_answer_fn(db, rt, user, kb.id, "guardrail-on"), True),
        }
        reports, spans = run_links(goldenset, links)
        lines.extend(guardrail_lines(reports["压缩前"], reports["压缩后"]))
        lines.extend(render_compare(
            reports, spans,
            note="两条**配置**跑的是同一条链路，只有压缩开关不同 —— 差异归因才干净。"))
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
