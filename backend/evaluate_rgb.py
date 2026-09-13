"""RGB 中文四能力评测：把官方数据建索引 → 走真实问答管线 → 按能力出数字。

指标本身在评测核心 app/eval_core.py（`run_eval` 按 `group` 出分能力表）；
本脚本只负责「按能力建索引 + 跑问答 + 落盘」。**不依赖运行中的服务**：本地 build_runtime()。

用法:  cd backend && python evaluate_rgb.py
数据:  从 https://github.com/chen700564/RGB 的 data/ 取 zh.json / zh_int.json / zh_fact.json，
       放到 backend/data/rgb/ 下（可用 EVAL_RGB_DIR 改）。缺哪份就报哪份 —— 不造数字。
报告:  logs/rgb-eval-report.log
环境:  EVAL_RGB_DIR 数据目录；EVAL_RGB_LIMIT 每种能力最多跑几条（默认 50，官方全量较慢）
"""
from __future__ import annotations

import os
import time

from app.eval_core import embedding_label, run_eval
from app.eval_index import index_chunks
from app.eval_rgb import ABILITIES, ability_label, available, load_entries

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BACKEND, "logs", "rgb-eval-report.log")
OWNER_ID = "rgb_eval_owner"
DATA_DIR = os.environ.get("EVAL_RGB_DIR", os.path.join(BACKEND, "data", "rgb"))
LIMIT = int(os.environ.get("EVAL_RGB_LIMIT", "50"))

_MISSING = ("请从 https://github.com/chen700564/RGB 的 data/ 取 zh.json / zh_int.json / "
            "zh_fact.json 放到该目录后再跑 —— 这里不会用假数据顶替。")


def _eval_user(db):
    """评测要在真实问答管线里跑，得有个发起人 —— 本地库建一个固定的评测用户。"""
    from app.eval_setup import eval_user

    return eval_user(db, "__rgb_eval__")


def _index_entry(rt, entry: dict, kb_id: str, tag: str) -> int:
    """把这条的官方文档切块 → 嵌入 → 进内存索引，返回块数。

    `tag` 让每条条目的块 id 互不相同（不同条目共用同一个向量库）。
    """
    chunks = []
    for d, doc in enumerate(entry["documents"]):
        doc_id = "%s_d%d" % (tag, d)
        for c in rt.chunker.chunk(doc, doc_id=doc_id, page_num=1):
            if c["chunk_type"] != "child":
                continue
            chunks.append({"id": c["id"], "content": c["content"],
                           "metadata": {"kb_id": kb_id, "owner_id": OWNER_ID, "doc_id": doc_id,
                                        "doc_name": "官方文档%d" % (d + 1), "page_num": 1,
                                        "content": c["content"]}})
    return index_chunks(rt, chunks)


def _answer_cursor(entries: list, kbs: list, answer_with):
    """按**条目顺序**取这一条自己的库，返回 core 要的 answer_fn。

    为什么不能按问题名建映射：噪声鲁棒与否定拒绝用的是同一批问题（同一个 zh.json），
    按问题名建映射会让后者覆盖前者，噪声那条就跑到"只有噪声文档"的库上了。
    顺序对不上直接抛 —— 绝不静默答错。
    """
    cursor = {"i": 0}

    def ask(question: str) -> dict:
        i = cursor["i"]
        cursor["i"] += 1
        if i >= len(entries) or entries[i]["question"] != question:
            raise RuntimeError("问答函数与条目顺序对不上（第 %d 条：期望 %r，实际 %r）"
                               % (i, entries[i]["question"] if i < len(entries) else None, question))
        return answer_with(kbs[i], question)

    return ask


def _write(lines) -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)


def main() -> None:
    avail = available(DATA_DIR)
    lines = ["=== RGB 中文四能力评测 ===",
             "数据目录: %s" % DATA_DIR,
             "嵌入: %s" % embedding_label(),
             "每种能力上限: %d 条" % LIMIT]

    if not any(avail.values()):
        lines += ["", "没有找到任何 RGB 官方数据 —— 一种能力都跑不了。", _MISSING]
        _write(lines)
        return

    for ability, ok in avail.items():
        if not ok:
            lines.append("跳过「%s」：缺少 %s" % (ability_label(ability), ABILITIES[ability]["file"]))

    entries: list = []
    for ability in ABILITIES:
        if avail[ability]:
            got = load_entries(DATA_DIR, ability)
            if not got:
                lines.append("「%s」文件在、但没有可用条目（当作没跑，不是 0 分）" % ability_label(ability))
                continue
            lines.append("载入「%s」%d 条（取前 %d）" % (ability_label(ability), len(got), LIMIT))
            entries.extend(got[:LIMIT])
    lines.append("")
    lines.append("口径：官方 answer 常是**多值**（zh_int.json 实测 100/100 条如此），而评测核心的")
    lines.append("      expect 是单个字符串 —— 这里**只核第一个值**，属于偏宽松的口径，别当成全核过了。")
    lines.append("")

    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.services import chat_service

    rt = build_runtime()
    kbs: list = []
    t0 = time.time()
    for i, e in enumerate(entries):
        kbs.append("rgbeval%d" % i)
        _index_entry(rt, e, kbs[-1], "rgb%d" % i)
    lines.append("建索引耗时 %.1fs（%d 条）" % (time.time() - t0, len(entries)))
    lines.append("")

    db = SessionLocal()
    try:
        user = _eval_user(db)

        def answer_with(kb_id: str, question: str) -> dict:
            out = chat_service.answer(db, rt, user, kb_id, question)
            return {"answer": out.get("answer", ""), "sources": out.get("sources", [])}

        lines.extend(run_eval(entries, _answer_cursor(entries, kbs, answer_with)).to_lines())
    finally:
        db.close()

    missing = [ability_label(a) for a, ok in avail.items() if not ok]
    if missing:
        lines += ["", "未跑的能力（数据缺失，不是 0 分）：%s" % " / ".join(missing)]
    _write(lines)


if __name__ == "__main__":
    main()
