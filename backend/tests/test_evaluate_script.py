"""evaluate.py 的职责边界（票 01 / #8）：只做 HTTP 管线与落盘，判据一律交给评测核心。

用假的 httpx.Client 照剧本应答，整条脚本离线跑通。
"""
from __future__ import annotations

import json

import evaluate


# 真实服务的每条流末尾都发它 —— 假对象要像真的（评测靠它判断「这条流跑完了」）
DONE = chr(10) + chr(10) + "data: [DONE]" + chr(10) + chr(10)


class _Resp:
    def __init__(self, payload=None, text="", status_code=200):
        self._payload = payload
        self.text = text
        self.status_code = status_code        # 假对象要像真的：产品代码会看状态码（#64 批 4）

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeClient:
    """只会照剧本应答的假 httpx.Client。"""

    def __init__(self, *args, **kwargs):
        pass

    def post(self, url, headers=None, json=None, files=None):
        if url.endswith("/auth/login"):
            return _Resp({"access_token": "t"})
        if url.endswith("/chat/stream"):
            return _Resp(text='data: {"type": "delta", "text": "甲"}' + DONE)
        if url.startswith("/api/v1/knowledge"):
            return _Resp({"id": "kb1"})
        if url.startswith("/api/v1/documents"):
            return _Resp({"id": "doc1"})
        return _Resp({})

    def get(self, url, headers=None):
        return _Resp({"status": "indexed", "chunk_count": 1, "page_count": 1})

    def delete(self, url, headers=None):
        return _Resp({"ok": True})

    def close(self):
        pass


# ---------- SSE 解析 ----------

def test_parse_sse_collects_answer_and_sources():
    body = "\n".join([
        'data: {"type": "sources", "data": [{"text": "甲", "page": 1}]}',
        "",
        'data: {"type": "delta", "text": "答"}',
        'data: {"type": "delta", "text": "案"}',
        "",
        'data: {"type": "done"}',
    ])
    got = evaluate._parse_sse(body)
    assert got == {"answer": "答案", "sources": [{"text": "甲", "page": 1}],
                   "citation_coverage": None, "context": None,
                   # 服务端没给 rerank 字段 = **不知道**，不是「没降级」
                   "rerank_degraded": None, "rerank_unscored": 0}


def test_parse_sse_ignores_malformed_and_non_data_lines():
    assert evaluate._parse_sse("data: not-json\n\n: keep-alive\n") == {
        "answer": "", "sources": [], "citation_coverage": None, "context": None,
        "rerank_degraded": None, "rerank_unscored": 0}


def test_answer_fn_asks_the_running_service():
    class C:
        def __init__(self):
            self.calls = []

        def post(self, url, headers=None, json=None):
            self.calls.append((url, json))
            return _Resp(text='data: {"type": "delta", "text": "甲"}' + DONE)

    c = C()
    assert evaluate._answer_fn(c, "kb1", {})("问")["answer"] == "甲"
    assert c.calls == [("/api/v1/chat/stream", {"kb_id": "kb1", "question": "问", "stream": True})]


# ---------- 整条脚本：判据来自核心 ----------

def test_main_runs_offline_and_judges_via_the_core(tmp_path, monkeypatch):
    """离线跑通 main()：脚本自己没有判据，期望事实命中一律由核心算出。"""
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": "Q", "expect": "甲"}]), encoding="utf-8")
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4")
    report = tmp_path / "eval-report.log"

    monkeypatch.setattr(evaluate, "GOLDEN", str(golden))
    monkeypatch.setattr(evaluate, "DOC", str(doc))
    monkeypatch.setattr(evaluate, "REPORT", str(report))
    monkeypatch.setattr(evaluate.httpx, "Client", _FakeClient)

    seen = {}
    real_run_eval = evaluate.run_eval

    def spy(goldenset, answer_fn, judge_fn=None, judge_label=None):
        seen["golden"] = list(goldenset)
        seen["answer"] = answer_fn("Q")
        seen["judge_label"] = judge_label
        return real_run_eval(goldenset, answer_fn, judge_fn, judge_label)

    monkeypatch.setattr(evaluate, "run_eval", spy)
    evaluate.main()

    assert seen["golden"] == [{"question": "Q", "expect": "甲"}]
    assert seen["answer"]["answer"] == "甲"          # answer_fn 真的打到服务（假 client）

    text = report.read_text(encoding="utf-8")
    assert "答案含期望事实 100%" in text
    assert "口径" in text                             # 报告带口径
    assert "上传: indexed" in text                     # 头部信息也在
    assert "=== RAGAS 四项 ===" in text                 # RAGAS 段无论有没有裁判都要出


def test_the_multi_turn_run_keeps_every_question_in_one_session():
    """多轮跑法才有历史可压 —— 所有问题必须落在同一个会话里（单轮则不带会话 id）。"""
    class C:
        def __init__(self):
            self.calls = []

        def post(self, url, headers=None, json=None):
            self.calls.append(json)
            return _Resp(text='data: {"type": "delta", "text": "甲"}' + DONE)

    c = C()
    evaluate._answer_fn(c, "kb1", {}, "eval-multi-turn")("问")
    assert c.calls[0]["session_id"] == "eval-multi-turn"

    c2 = C()
    evaluate._answer_fn(c2, "kb1", {})("问")
    assert "session_id" not in c2.calls[0]


def test_parse_sse_notices_a_degraded_rerank():
    """降级不打断回答（对），但报告要知道这次没重排 —— 别把 RRF 顺序的数字说成「重排已跑」（#48）。"""
    body = "\n".join([
        'data: {"type": "delta", "text": "答"}',
        "",
        'data: {"type": "done", "rerank": {"degraded": true, "note": "不可达"}}',
    ])
    assert evaluate._parse_sse(body)["rerank_degraded"] is True


def test_parse_sse_treats_a_healthy_rerank_as_not_degraded():
    body = "\n".join([
        'data: {"type": "delta", "text": "答"}',
        "",
        'data: {"type": "done", "rerank": {"degraded": false, "note": ""}}',
    ])
    assert evaluate._parse_sse(body)["rerank_degraded"] is False


def test_a_failed_upload_says_why_in_the_report():
    """报告写「上传: failed」却不说原因，等于把排查的第一步留给读者去猜。

    实测就卡在这儿：真机上跑出来 `上传: failed chunks=0 页数=46`，而失败原因
    （`_upload_line` 拿得到 `error` 字段）根本没打出来。
    """
    line = evaluate._upload_line({"status": "failed", "chunk_count": 0, "page_count": 46,
                                  "error": "解析失败: 字体表缺失"})
    assert "上传: failed" in line and "解析失败: 字体表缺失" in line


def test_a_healthy_upload_stays_terse():
    line = evaluate._upload_line({"status": "indexed", "chunk_count": 440, "page_count": 46,
                                  "error": ""})
    assert line == "上传: indexed chunks=440 页数=46"


def test_a_upload_that_did_not_land_aborts_the_section(monkeypatch):
    """文档没入库就不能往下算 —— 那 10 题的「事实命中 100%」是模型拿自己知识答出来的。

    实测踩过：`上传: failed chunks=0`，但报告照样给出 100% 命中；后三题来源为空、
    答案里还写着 `[来源: 已知信息]`（来自跨会话记忆）。缺前置一律写「未跑」。
    """
    import pytest

    class C:
        def post(self, url, headers=None, json=None, files=None):
            if url.endswith("/auth/login"):
                return _Resp({"access_token": "t"})
            if url.startswith("/api/v1/knowledge"):
                return _Resp({"id": "kb1"})
            return _Resp({"id": "doc1"})

        def get(self, url, headers=None):
            return _Resp({"status": "failed", "chunk_count": 0, "page_count": 46,
                          "error": "解析失败: 字体表缺失"})

        def close(self):
            pass

    import evaluate as ev
    monkeypatch.setattr(ev.httpx, "Client", lambda *a, **k: C())

    with pytest.raises(RuntimeError) as e:
        ev.run_online()

    assert "没入库" in str(e.value) and "字体表缺失" in str(e.value)


def test_a_stream_that_breaks_mid_way_is_not_counted_as_a_miss():
    """状态码 200、但**流中途断了**（比如生成模型没权限）—— 那一次不作数，不能记成「没命中」。

    实测踩过：生成模型换成 403 的模型后，每问都以 200 开头、吐完 sources 就断，
    `_parse_sse` 只拿到 answer=""，报告会印出一个**假的 0%** —— 与 #64 批 4 修的那条同一类。
    """
    import pytest

    class C:
        def post(self, url, headers=None, json=None):
            return _Resp(text='data: {"type": "sources", "data": []}')     # 没有 [DONE]

    with pytest.raises(RuntimeError) as e:
        evaluate._answer_fn(C(), "kb1", {})("问")

    assert "没跑完" in str(e.value)


def test_a_complete_stream_still_parses_normally():
    class C:
        def post(self, url, headers=None, json=None):
            return _Resp(text='data: {"type": "delta", "text": "甲"}' + DONE)

    assert evaluate._answer_fn(C(), "kb1", {})("问")["answer"] == "甲"
