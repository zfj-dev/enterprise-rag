"""跨会话记忆（票 26）：一份可复现的**召回示例** + 引用覆盖率护栏。

一、跨会话示例：会话 A 里用户告知一件事 → **异步抽取**落库 → 会话 B（**新会话**）召回并注入
    prompt → 模型据此作答。示例走**真链路**（抽取 / 落库 / 召回 / 注入都真的跑）。
    演示模式（假模型）下 LLM 换成**确定性替身**，离线可复现、不必有 API Key；真实模式下用
    真模型（那正是不必假装的东西）。替身只做两件事：恒等改写问题、**只在 prompt 里有已知信息时**
    才据它作答 —— 所以示例证明的是「那条事实进了模型看到的 prompt」，不是「模型答得对」。
二、护栏：同一黄金集跑两遍，只有记忆开关不同。判的是**答案侧的引用覆盖率**（记忆真能推动的量），
    来源侧指标因「记忆不进入 sources」是构造性不变的 —— 详见 eval_compare.memory_guardrail_lines。

用法:  cd backend && python evaluate_memory.py   （或 scripts/evaluate_memory.ps1）
环境:  EVAL_GOLDEN / EVAL_DOC 同 evaluate.py；报告路径 EVAL_MEMORY_REPORT
       EVAL_MEMORY_SEED  额外播一条记忆（分号分隔）—— 想让护栏真正压到「记忆被注入」那一路，
                         就给一条与黄金集相关的；否则零注入，护栏只回归「行为与今日一致」
报告:  logs/memory-cross-session.log（助手可读）
"""
from __future__ import annotations

import json
import os
import time

from app.eval_compare import memory_guardrail_lines, render_compare, run_links
from app.eval_core import Report
from app.eval_setup import (drop_kb, ensure_schema, eval_user, ingest_file, new_kb,
                            with_setting, write_report)

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.environ.get("EVAL_MEMORY_REPORT",
                        os.path.join(BACKEND, "logs", "memory-cross-session.log"))
GOLDEN = os.environ.get("EVAL_GOLDEN", os.path.join(BACKEND, "data", "golden_set_paper.json"))
DOC = os.environ.get("EVAL_DOC", os.path.join(BACKEND, "paper.pdf"))
USERNAME = "__memory_eval__"

# 示例的三句话：告知 → 落库的事实 → 新会话里的追问。事实与追问刻意共享关键词，召回才命中。
TELL = "我在跟电池项目，口径按季度统计"
FACT = "用户在跟电池项目，口径按季度统计"
ASK = "电池项目的口径是什么？"
MEMORY_ANSWER = "（已读到已知信息）"
MEMORY_HEADER = "【已知信息】"


class DemoLLM:
    """确定性替身模型：**恒等改写**问题，且**只**在 prompt 里有已知信息时才据它作答。

    这样「会话 B 的回应取决于那条记忆有没有到 prompt」就是可复现的。
    `is_fake=False` —— 链路因此走真实分支（改写 / 引用校验 / **异步抽取**都真的跑），
    演示要展示的正是这些环节（假模型会把抽取整段跳过）。
    """

    is_fake = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, messages):
        content = messages[-1]["content"]
        self.prompts.append(content)
        if "查询改写助手" in content:          # 恒等改写：保持检索 / 召回用的查询带关键词
            yield content.rsplit("当前问题：", 1)[-1].strip()
        elif MEMORY_HEADER in content:
            yield "%s%s" % (MEMORY_ANSWER, FACT)
        else:
            yield "（没有已知信息）"


class DemoExtractor:
    """确定性抽取器：把用户那句**明确告知**抽出来。

    真实模式由 LLM 抽（`LlmFactExtractor`）——演示模式的假模型默认不抽（它的输出不是用户陈述）。
    """

    def extract(self, question: str, answer: str) -> list[str]:
        return [FACT] if TELL in question else []


class NullExtractor:
    """护栏两次跑都关掉**抽取**：只留召回这一个变量，两列才可比。

    否则记忆开启那一列会在跑的过程中不断抽出新事实、后面几问看到的记忆比关闭那列多 ——
    差异就不只来自「开关」了。
    """

    def extract(self, question: str, answer: str) -> list[str]:
        return []


def _prompt_had_memory(llm, since: int) -> str:
    """会话 B 那几次模型调用里，有没有一次真看到已知信息块。"""
    prompts = getattr(llm, "prompts", None)
    if prompts is None:
        return "（真实模型：prompt 未记录）"
    return "是" if any(MEMORY_HEADER in p for p in prompts[since:]) else "否"


def _wait_for_facts(store, user_id: str, known: set, timeout: float = 5.0) -> list[dict]:
    """等异步抽取落库（抽取在后台线程里跑，示例得等它一下）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = [f for f in store.list(user_id) if f["id"] not in known]
        if new:
            return new
        time.sleep(0.05)
    return []


def _demo_lines(db, rt, user, kb_id: str) -> list[str]:
    """跑一遍跨会话示例，返回报告段落（成不成写在里面，不预先下结论）。"""
    from app.services import chat_service

    out = ["", "=== 一、跨会话召回示例（会话 A 告知 → 会话 B 在新会话里追问）===",
           "会话 A（告知）: %s" % TELL,
           "会话 B（新会话）: %s" % ASK,
           ""]
    facts_before = {f["id"] for f in rt.memory_store.list(user.id)}

    fake = getattr(rt.llm, "is_fake", False)
    llm = DemoLLM() if fake else rt.llm
    saved_llm, saved_extractor = rt.llm, rt.fact_extractor_factory
    rt.llm = llm
    rt.fact_extractor_factory = lambda _llm: DemoExtractor()
    try:
        chat_service.answer(db, rt, user, kb_id, TELL, "demo-session-a")
        landed = _wait_for_facts(rt.memory_store, user.id, facts_before)
        out.append("会话 A 说完 → 异步抽取落库 %d 条：" % len(landed))
        out.extend("  - %s" % f["content"] for f in landed)
        if not landed:
            out.append("  （没等到落库 —— 抽取是异步旁路，这一步失败不该影响回答）")

        since = len(getattr(llm, "prompts", []) or [])
        got_b = chat_service.answer(db, rt, user, kb_id, ASK, "demo-session-b")
        trace = got_b.get("trace") or {}
        sources = got_b.get("sources") or []
        recalled = trace.get("memory_recalled") or 0
        in_prompt = _prompt_had_memory(llm, since)
        leaked = any(FACT in str(s.get("text") or "") for s in sources)

        out.append("")
        out.append("会话 B 的召回: %s 条" % recalled)
        out.append("  注入形态: 独立的 %s 块（与【参考资料】分开，不占检索候选）" % MEMORY_HEADER)
        out.append("  模型看到的 prompt 里含 %s: %s" % (MEMORY_HEADER, in_prompt))
        out.append("  会话 B 的回答: %s" % (got_b.get("answer") or "").strip())
        out.append("  硬约束核验（实算，不是照抄句子）：记忆混进 sources 了吗 —— %s"
                   % ("**是 —— 违反！**" if leaked else "没有"))
        out.append("    会话 B 的 sources 共 %d 条，回答依据仍来自文档" % len(sources))
        out.append("")
        if recalled and in_prompt == "是":
            out.append("  判读：回应里出现那条事实，说明**会话 A 告知的东西在新会话里生效了**；"
                       "而它不是来源 —— 记忆只用于理解问题与个性化，不作答依据。")
        else:
            out.append("  判读：**没看到记忆进入 prompt**（召回 %s 条）—— 这一段没跑成，"
                       "别当成演示成功。" % recalled)
    finally:
        rt.llm, rt.fact_extractor_factory = saved_llm, saved_extractor
    return out


def _seed_extra(rt, user_id: str) -> list[str]:
    """EVAL_MEMORY_SEED：额外播一条记忆，用来让护栏压到「记忆被注入」那一路。"""
    facts = [t.strip() for t in os.environ.get("EVAL_MEMORY_SEED", "").split(";") if t.strip()]
    if facts:
        rt.memory_store.add(user_id, facts, session_id="eval-seed")
    return facts


def _drop_memory(rt, user_id: str) -> None:
    """清掉这次评测留下的记忆 —— 不清的话每跑一次就多堆一批，召回数与护栏都会漂。"""
    try:
        for f in rt.memory_store.list(user_id):
            rt.memory_store.delete(user_id, f["id"])
    except Exception as e:      # noqa: BLE001 —— 清理失败不该毁掉已算出的报告
        print("清记忆失败（不影响报告）：%s" % e)


def _recall_count(rt, user_id: str, goldenset) -> tuple[int, int]:
    """数一遍「当前记忆对黄金集问题会召回几条」+ 召回失败的条数。

    用**原问题**：改写是链路内部的事，这里只想量记忆库与问题的相关性。
    失败**计数**而不是当成 0 —— 「召回挂了」和「确实没相关的」是两回事，不许混成一句。
    """
    total = failed = 0
    for item in goldenset:
        try:
            total += len(rt.recall_memory(user_id, item.get("question", "")))
        except Exception:      # noqa: BLE001 —— 与链路的旁路口径一致，但要报出来
            failed += 1
    return total, failed


def _identical(before: Report, after: Report) -> bool:
    """两份报告是否**逐项**一致（零注入时的回归判据）—— 只比评测核心认的那些事实。"""
    if len(before.items) != len(after.items):
        return False
    return all((b.fact_hit, b.grounded, b.page_hit, b.answer)
               == (a.fact_hit, a.grounded, a.page_hit, a.answer)
               for b, a in zip(before.items, after.items))


def _config_lines() -> list[str]:
    from app.config import get_settings

    s = get_settings()
    return [
        "模型: LLM_PROVIDER=%s model=%s" % (s.llm_provider, s.llm_model),
        "记忆: memory_enabled=%s；召回 top-k=%d，相似度阈值 %s，单条注入上限 %d 字"
        % (s.memory_enabled, s.memory_recall_top_k, s.memory_recall_min_score,
           s.memory_inject_max_chars),
    ]


def main(golden: str | None = None, doc: str | None = None, report: str | None = None) -> None:
    golden = golden or GOLDEN
    doc = doc or DOC
    report = report or REPORT
    lines = ["=== 跨会话记忆：召回示例 + 引用覆盖率护栏 ===",
             "黄金集: %s" % golden,
             "文档: %s" % doc,
             "报告: %s" % report]
    lines += _config_lines()

    if not os.path.exists(golden) or not os.path.exists(doc):
        lines += ["", "未跑：缺输入。",
                  "  黄金集 %s：%s" % (golden, "有" if os.path.exists(golden) else "**缺**"),
                  "  文档 %s：%s" % (doc, "有" if os.path.exists(doc) else "**缺**"),
                  "补上前置再跑 —— 这里不会拿假数据顶替。"]
        write_report(report, lines)
        return

    with open(golden, encoding="utf-8") as f:
        goldenset = json.load(f)

    from app.config import get_settings
    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.eval_agent import deterministic_answer_fn

    ensure_schema()
    rt = build_runtime()
    db = SessionLocal()
    kb = None
    try:
        user = eval_user(db, USERNAME)
        _drop_memory(rt, user.id)          # 从干净的记忆开始，报告才可复现
        kb = new_kb(db, user.id, "跨会话记忆评测库")
        doc_id = ingest_file(db, rt, doc, user.id, kb.id)
        lines.append("")
        lines.append("文档已入库：doc_id=%s" % doc_id)
        if get_settings().llm_provider == "fake":
            lines.append("注意：LLM_PROVIDER=fake（演示模式）—— 二、护栏那一节两列答案都是固定文本，"
                         "答案侧的覆盖率天然一致，护栏要在真实模型下才有信息量。"
                         "一、示例照跑：它用确定性替身走真链路。")
        lines.extend(_demo_lines(db, rt, user, kb.id))

        seeded = _seed_extra(rt, user.id)
        if seeded:
            lines.append("")
            lines.append("EVAL_MEMORY_SEED 额外播入 %d 条记忆：%s" % (len(seeded), " / ".join(seeded)))

        recalled, recall_failed = _recall_count(rt, user.id, goldenset)
        saved_extractor = rt.fact_extractor_factory
        rt.fact_extractor_factory = lambda _llm: NullExtractor()   # 只留召回这一个变量
        try:
            links = {
                "记忆关闭": with_setting(
                    deterministic_answer_fn(db, rt, user, kb.id, "memory-off"),
                    "memory_enabled", False),
                "记忆开启": with_setting(
                    deterministic_answer_fn(db, rt, user, kb.id, "memory-on"),
                    "memory_enabled", True),
            }
            reports, spans = run_links(goldenset, links)
        finally:
            rt.fact_extractor_factory = saved_extractor

        lines.extend(memory_guardrail_lines(reports["记忆关闭"], reports["记忆开启"]))
        lines.append("  本次记忆召回：按黄金集**原问题**计共 %d 条（%d 条召回失败）"
                     % (recalled, recall_failed))
        if not recalled:
            lines.append("    零注入 —— 两列**逐项**一致：%s"
                         % ("是（「无相关记忆时行为与今日一致」成立）"
                            if _identical(reports["记忆关闭"], reports["记忆开启"])
                            else "**否 —— 零注入之下仍有差异，得查**"))
        lines.extend(render_compare(
            reports, spans,
            note="两列跑的是同一条链路（本地确定性管线），只有记忆开关不同；抽取在护栏里关掉了，"
                 "所以差异只来自**召回与注入**。"))
    finally:
        _drop_memory(rt, user.id)
        if kb is not None:
            try:
                drop_kb(db, kb.id)
            except Exception as e:      # noqa: BLE001 —— 清理失败不该毁掉已算出的报告
                print("清库失败（不影响报告）：%s" % e)
        db.close()
    write_report(report, lines)


if __name__ == "__main__":
    main()
