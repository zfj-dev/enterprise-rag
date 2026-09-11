"""一页评测报告：一条命令跑完所有已实现的指标，落成一份可对比的固定报告。

各段的**口径与实现都在各自的脚本里**，本文件不重复实现任何指标 —— 只负责
跑 → 收 → 拼成一页 → 落盘。缺前置条件的段**明说没跑**，绝不产出假数字。

段与其前置条件：
  生成层 + RAGAS   打运行中的服务（evaluate.run_online）
  检索层            离线（evaluate_retrieval）
  RGB 中文四能力    离线，需要官方数据（evaluate_rgb）
  延迟（并发）      打运行中的服务（evaluate_latency）

用法:  python backend/evaluate_all.py
报告:  logs/eval-summary.log（固定路径，便于和历史报告比涨退）
环境:  同各分段脚本；另可用 EVAL_SKIP=latency,rgb 跳过较慢的段
       （段 key：generation / retrieval / rgb / latency，按段名中文也认）
"""
from __future__ import annotations

import os
from datetime import datetime

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BACKEND, "logs", "eval-summary.log")

# 只作参照的目标线 —— 不达标只标一下，不卡任何流程（spec 0001 明确「不引入 CI 发版门禁」）。
# (key, 显示名, 阈值, 从报告里取值)：取值函数直接绑在表里，别再拿显示名当判别键 ——
# 那样新增一条会静默落到 else 分支、报出别人的数字。
TARGETS = (
    ("fact", "答案含期望事实", 0.80, lambda r: r.fact_rate),
    ("grounded", "引用忠实度", 0.80, lambda r: r.grounded_rate),
    ("refusal", "拒答率（负样本）", 0.80, lambda r: r.refuse_rate),
    ("faithfulness", "RAGAS 忠实度", 0.80, lambda r: (r.ragas or {}).get("faithfulness")),
)


def _config_snapshot() -> list:
    """配置快照：模型 / 开关 / 关键参数 —— 没有它，两份报告的数字没法比。"""
    from app.config import get_settings

    s = get_settings()
    keys = ("use_real", "llm_provider", "llm_model", "embedding_provider", "embedding_model",
            "embedding_device", "reranker_provider", "reranker_device", "vector_store",
            "redis_url", "parser_use_docling", "semantic_cache", "memory_enabled",
            "context_compress", "context_token_budget", "retrieval_top_k", "rerank_top_k",
            "rrf_k", "min_relevance", "chunk_child_size", "chunk_parent_size",
            "ragas_judge_model", "ragas_judge_temperature")
    marker = object()
    out, absent = [], []
    for k in keys:
        v = getattr(s, k, marker)
        if v is marker:
            absent.append(k)            # 键没了就说出来（改名了会悄悄少一项，快照就不可比）
        else:
            out.append("  %-24s = %s" % (k, v))
    if absent:
        out.append("  （这些键在 settings 里不存在，可能已改名：%s）" % ", ".join(absent))
    return out


def _target_lines(report) -> list:
    out = ["  数字取自本页「生成层指标」；不达标只是标注，不影响任何流程"]
    for _key, name, threshold, get in TARGETS:
        if report is None:
            out.append("  %-18s ≥ %.0f%%   （本次没跑生成层）" % (name, threshold * 100))
            continue
        value = get(report)
        if value is None:
            out.append("  %-18s ≥ %.0f%%   本次没有这类数字" % (name, threshold * 100))
        else:
            mark = "[达标]" if value >= threshold else "[未达标]"
            out.append("  %-18s ≥ %.0f%%   实际 %.0f%%  %s"
                       % (name, threshold * 100, value * 100, mark))
    return out


def _section(name: str, module, log_path: str) -> list:
    """跑的是一段独立脚本，收的是它自己落盘的报告 —— 口径只在那一处。"""
    out = ["", "=== %s ===" % name]
    try:
        module.main()
    except Exception as e:   # noqa: BLE001 —— 这段没跑成要明说，不能当没这回事
        out.append("未跑：%s: %s" % (type(e).__name__, e))
        return out
    if not os.path.isfile(log_path):
        out.append("跑完了但没有落盘报告（%s）" % log_path)
        return out
    with open(log_path, encoding="utf-8") as f:
        out.extend(f.read().splitlines())
    return out


def main() -> None:
    import evaluate
    import evaluate_latency
    import evaluate_retrieval
    import evaluate_rgb

    skip = {s.strip() for s in os.environ.get("EVAL_SKIP", "").split(",") if s.strip()}
    lines = ["=== RAG 评测一页报告 ===",
             "生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "",
             "=== 配置快照 ==="]
    lines.extend(_config_snapshot())

    report = None
    if "generation" in skip or "生成层" in skip:
        gen = ["", "=== 生成层指标 ===", "按 EVAL_SKIP 跳过"]
    else:
        try:
            report, upload = evaluate.run_online()
            gen = ["", "=== 生成层指标 ===",
                   "黄金集: %s" % evaluate.GOLDEN, "被评文档: %s" % evaluate.DOC,
                   evaluate._upload_line(upload),      # 同一份格式只在 evaluate.py 里写
                   ""]
            gen.extend(report.to_lines())
        except Exception as e:   # noqa: BLE001 —— 服务没起也不该让整页报告消失
            gen = ["", "=== 生成层指标 ===",
                   "未跑：%s: %s（生成层与延迟都要服务在跑）" % (type(e).__name__, e)]

    lines += ["", "=== 目标线（只作参照，不卡发版）==="]
    lines.extend(_target_lines(report))
    lines += gen

    for key, name, module, log_path in (
        ("retrieval", "检索层（离线）", evaluate_retrieval, evaluate_retrieval.REPORT),
        ("rgb", "RGB 中文四能力（离线）", evaluate_rgb, evaluate_rgb.REPORT),
        ("latency", "延迟（并发）", evaluate_latency, evaluate_latency.REPORT),
    ):
        if key in skip or name in skip or name.split("（")[0] in skip:
            lines += ["", "=== %s ===" % name, "按 EVAL_SKIP 跳过"]
            continue
        lines.extend(_section(name, module, log_path))

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
