"""工具结果清理（票 21 / #28）：**已被引用**的工具结果收缩为元信息，未引用的留到本轮结束。

缝：进程内传输 + 脚本化 LLM —— 确定性、无网络、不起真 MCP 进程。
收缩只丢"全文"，chunk_id / 文档名 / 页码 / 首部片段一个不少 —— 引用仍可回溯到原文片段。
"""
from __future__ import annotations

from app.core.agent import run_agent
from app.core.context import cited_chunk_ids, trim_tool_result
from app.core.llm import ToolCall
from app.core.tools import CALCULATOR, Tool, ToolRegistry
from app.mcp.client import InProcessTransport

LONG = "比亚迪2025年营业收入为803.96亿元，同比增长25%。" * 20   # 远超首部片段上限


class ScriptedLLM:
    """按脚本吐回复（用完吐终答）。记录每次调用，并**持有 messages 本体**——
    轮末收尾是在最后一次模型调用之后做的，只有拿住那个 list 才观察得到。"""

    is_fake = False

    def __init__(self, replies, fallback="（收敛作答）"):
        self.replies = list(replies)
        self.fallback = fallback
        self.calls: list[dict] = []
        self.live: list | None = None

    def stream(self, messages):
        yield self.fallback

    def chat_with_tools(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        self.live = messages
        if not tools:                                   # 收敛那一步：不给工具
            return {"content": self.fallback, "tool_calls": []}
        if self.replies:
            return self.replies.pop(0)
        return {"content": self.fallback, "tool_calls": []}


def _src(cid, doc="年报.pdf", page=3, text=LONG):
    return {"chunk_id": cid, "doc_id": "d1", "doc_name": doc, "page": page,
            "text": text, "score": 0.9}


def _kb_tool(sources):
    return Tool(name="KbRetrieve", description="检索",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: {"sources": sources})


def _kb_call(cid="k1"):
    return ToolCall(id=cid, name="KbRetrieve", arguments={"query": "营收"})


def _calc_call(cid="c1"):
    return ToolCall(id=cid, name="Calculator", arguments={"expression": "1+1"})


def _transport(sources):
    return InProcessTransport(ToolRegistry(tools=[_kb_tool(sources), CALCULATOR]))


def _tool_msg(messages, tool_call_id):
    """捞出某次调用里，某条工具结果消息的正文。"""
    hits = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id]
    assert hits, "这次调用里没有 tool_call_id=%s 的工具消息" % tool_call_id
    return hits[-1]["content"]


# ---------- 纯函数：只丢全文，不丢可回溯 ----------

def test_trim_drops_only_the_full_text():
    """收缩后 chunk_id / 文档名 / 页码 / 首部片段都在 —— 引用仍能定位到原文片段。"""
    out = trim_tool_result({"sources": [_src("ch1")]}, {"ch1"})

    s = out["sources"][0]
    assert (s["chunk_id"], s["doc_name"], s["page"]) == ("ch1", "年报.pdf", 3)
    assert s["text"] != LONG and LONG not in s["text"]
    assert "803.96" in s["text"]                 # 首部片段还是原文
    assert out["trimmed"] is True                # 标记是布尔不是散文：一个键，别把省下的又花回去


def test_trim_only_touches_cited_chunks():
    """同一结果里没被引用的块照旧保留全文 —— 未引用的内容不许提前收缩。"""
    other = "其他内容" * 60
    out = trim_tool_result({"sources": [_src("ch1"), _src("ch2", text=other)]}, {"ch1"})

    assert LONG not in out["sources"][0]["text"]     # 被引用的那块收缩了
    assert out["sources"][1]["text"] == other        # 没被引用的那块一个字没动


def test_nothing_to_trim_returns_none():
    """没被引用、没有 sources（Calculator / SqlQuery）、正文本来就短 —— 一律原样，不做无谓替换。"""
    assert trim_tool_result({"sources": [_src("ch1")]}, set()) is None
    assert trim_tool_result({"value": 2}, {"ch1"}) is None
    assert trim_tool_result({"sources": [_src("ch1", text="短")]}, {"ch1"}) is None
    assert trim_tool_result({"sources": []}, {"ch1"}) is None


def test_a_citation_is_recognised_by_marker_or_by_chunk_id():
    srcs = [_src("ch1"), _src("ch2", doc="季报.pdf", page=7)]
    assert cited_chunk_ids("[来源: 年报.pdf, 第3页] 是这样", srcs) == {"ch1"}
    assert cited_chunk_ids("见 [来源：季报.pdf, 第7页] 的表格", srcs) == {"ch2"}
    assert cited_chunk_ids("直接点名 chunk_id: ch2 这条", srcs) == {"ch2"}
    assert cited_chunk_ids("文档名对了但页码不对 [来源: 年报.pdf, 第9页]", srcs) == set()
    assert cited_chunk_ids("", srcs) == set()


# ---------- 循环里：什么时候收缩 ----------

def test_a_cited_tool_result_is_shrunk_before_the_next_step():
    """模型这条回复里引用过的结果，在**下一次**模型调用前就收缩掉 —— 预算不再花在已用掉的内容上。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "按资料，营收803.96亿元 [来源: 年报.pdf, 第3页]。再算个比例。",
         "tool_calls": [_calc_call()]},
        {"content": "算完了。", "tool_calls": []},
    ])
    got = run_agent("营收多少", llm=llm,
                    transport=_transport([_src("ch1"),
                                          _src("ch2", doc="季报.pdf", page=7, text="其他内容" * 60)]))

    fed = _tool_msg(llm.calls[2]["messages"], "k1")
    assert "ch1" in fed and "年报.pdf" in fed and '"page": 3' in fed
    assert LONG not in fed                               # 全文没了
    assert "其他内容" * 60 in fed                        # 没被引用的那块保留
    assert got["answer"] == "算完了。"


def test_the_shrink_does_not_rewrite_what_the_model_already_saw():
    """收缩是**换掉**那条消息，不是原地改 —— 模型上一次调用看到的全文快照不许被回头篡改。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "[来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},
        {"content": "算完了。", "tool_calls": []},
    ])
    run_agent("问", llm=llm, transport=_transport([_src("ch1")]))

    assert LONG in _tool_msg(llm.calls[1]["messages"], "k1")       # 第二次调用时还是全文
    assert LONG not in _tool_msg(llm.calls[2]["messages"], "k1")   # 第三次调用时已收缩


def test_an_uncited_tool_result_survives_the_round_and_is_only_tidied_at_the_end():
    """没人引用的结果**不许**提前收缩 —— 收敛作答那一步模型还得靠它；收尾发生在收敛之后。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "我再验一下别的。", "tool_calls": [_calc_call()]},   # 没引用任何来源
    ])
    got = run_agent("问", llm=llm, transport=_transport([_src("ch1")]), max_steps=2)

    assert LONG in _tool_msg(llm.calls[1]["messages"], "k1")       # 循环中：原样
    converged = llm.calls[2]                                        # 跑到上限后的收敛调用
    assert converged["tools"] is None
    assert LONG in _tool_msg(converged["messages"], "k1")           # 收敛作答时仍拿得到全文
    assert got["stopped"] == "max_steps"

    final = _tool_msg(llm.live, "k1")                               # 本轮结束后才收尾
    assert LONG not in final and "ch1" in final and '"page": 3' in final


def test_a_partially_cited_result_is_tidied_at_the_round_end_too():
    """只被引用了一部分的结果，剩下那几块轮末也得收干净 —— 不能因为收过一次就漏掉。"""
    other = "其他内容" * 60
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "[来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},   # 只引用了 ch1
    ])
    run_agent("问", llm=llm,
              transport=_transport([_src("ch1"), _src("ch2", doc="季报.pdf", page=7, text=other)]),
              max_steps=2)

    mid = _tool_msg(llm.calls[2]["messages"], "k1")       # 循环内：只收了被引用的那块
    assert LONG not in mid and other in mid
    final = _tool_msg(llm.live, "k1")                     # 轮末：剩下那块也收干净
    assert LONG not in final and other not in final
    assert "ch1" in final and "ch2" in final              # 两块都还在，锚点没丢


def test_multi_step_reasoning_still_completes_after_trimming():
    """回归：收缩不得把多步推理打断 —— 该答还得答出来。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "按资料 803.96 亿元 [来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},
        {"content": "最终答案：同比增长 25%。", "tool_calls": []},
    ])
    got = run_agent("算同比增长", llm=llm, transport=_transport([_src("ch1")]))

    assert got["stopped"] == "answered"
    assert got["answer"] == "最终答案：同比增长 25%。"
    assert [s["tool"] for s in got["steps"]] == ["KbRetrieve", "Calculator"]


def test_the_trim_can_be_switched_off():
    """关掉清理：行为与今天一致 —— 工具结果一个字都不动。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "[来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},
    ])
    run_agent("问", llm=llm, transport=_transport([_src("ch1")]),
              max_steps=2, trim_tool_results=False)

    assert LONG in _tool_msg(llm.calls[2]["messages"], "k1")


# ---------- 对外的一致性 ----------

def test_the_runs_sources_keep_the_full_text_for_the_citation_preview():
    """收缩只作用于**喂给模型的副本**：对外返回的 sources 仍是全文，前端预览能定位原文。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "[来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},
        {"content": "算完了。", "tool_calls": []},
    ])
    got = run_agent("问", llm=llm, transport=_transport([_src("ch1")]))

    assert got["sources"][0]["text"] == LONG
    assert got["sources"][0]["chunk_id"] == "ch1"


def test_the_trace_reports_how_many_results_were_shrunk():
    """收了几条要看得见 —— 不然"省了多少"又只能靠嘴说。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "[来源: 年报.pdf, 第3页]", "tool_calls": [_calc_call()]},
        {"content": "算完了。", "tool_calls": []},
    ])
    got = run_agent("问", llm=llm, transport=_transport([_src("ch1")]))

    assert got["trace"]["tool_results_trimmed"] == 1


def test_a_citation_in_the_final_answer_does_not_count_as_a_trim():
    """终答轮才被引用的结果，收了也没有下一次调用会读它 —— 计数只报真省下预算的那批。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_kb_call()]},
        {"content": "按资料 803.96 亿元 [来源: 年报.pdf, 第3页]", "tool_calls": []},   # 直接终答
    ])
    got = run_agent("营收多少", llm=llm, transport=_transport([_src("ch1")]))

    assert got["stopped"] == "answered"
    assert got["trace"]["tool_results_trimmed"] == 0


def test_a_run_without_retrieval_reports_zero_trims():
    """Calculator 结果没有全文，谈不上收缩 —— 计数如实为零。"""
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_calc_call()]},
        {"content": "等于 2", "tool_calls": []},
    ])
    got = run_agent("1+1", llm=llm, transport=_transport([]))

    assert got["trace"]["tool_results_trimmed"] == 0
