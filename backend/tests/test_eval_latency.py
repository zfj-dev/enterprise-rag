"""延迟分桶（票 07 / #14）：分段百分位、并发口径、与正确性分开。

样本全部注入 —— 不联网、不调模型，结果确定。
"""
from __future__ import annotations

from app.eval_core import LatencyMetrics, Report, percentile, run_eval
from tests.helpers import register_and_kb, sse_events


# ---------- 百分位 ----------

def test_percentile_uses_nearest_rank():
    xs = list(range(1, 101))                 # 1..100
    assert percentile(xs, 50) == 50
    assert percentile(xs, 95) == 95
    assert percentile(xs, 100) == 100


def test_percentile_on_a_single_sample():
    assert percentile([7], 50) == 7 and percentile([7], 95) == 7


def test_percentile_ignores_missing_and_empty():
    assert percentile([None, 5, None], 95) == 5
    assert percentile([], 50) is None


def test_percentile_never_puts_p95_below_p50():
    xs = [3, 1, 2]
    assert percentile(xs, 95) >= percentile(xs, 50)


# ---------- 报告 ----------

def _metrics():
    return LatencyMetrics(concurrent=4, rounds=1, note="测试口径", samples=[
        {"retrieval": 10, "rerank": 30, "ttft": 100, "generate": 800},
        {"retrieval": 20, "rerank": 60, "ttft": 200, "generate": 1600},
    ])


def test_every_stage_is_visible_with_its_own_percentiles():
    m = _metrics()
    assert m.count == 2
    assert m.p("retrieval", 50) == 10 and m.p("retrieval", 95) == 20
    assert m.p("rerank", 95) == 60
    assert m.p("generate", 50) == 800
    assert m.p("ttft", 50) == 100

    text = "\n".join(m.to_lines())
    assert "并发 4" in text and "测试口径" in text            # 并发数与测量方式都要写明
    for label in ("检索", "重排", "首字(TTFT)", "生成"):
        assert label in text
    assert "不参与正确性判定" in text                          # 与正确性分开
    assert "样本 n=2" in text


def test_missing_stage_prints_a_dash_not_a_zero():
    m = LatencyMetrics(concurrent=2, rounds=1, samples=[{"generate": 500}])
    assert m.p("retrieval", 50) is None
    assert "-" in "\n".join(m.to_lines())                   # 缺就是缺，不拿 0 冒充


def test_empty_samples_are_said_not_faked():
    m = LatencyMetrics(concurrent=4, rounds=1, samples=[])
    assert m.count == 0
    assert "没有采到样本" in "\n".join(m.to_lines())


def test_report_can_carry_latency_alongside_generation():
    """票 08 一页报告：延迟段与生成层数字在同一份里。"""
    gen = run_eval([{"question": "Q", "expect": "甲"}], lambda q: {"answer": "甲"})
    text = "\n".join(Report(items=gen.items, latency=_metrics()).to_lines())

    assert "=== 生成层指标 ===" in text
    assert "=== 延迟（并发 4 × 1 轮）===" in text
    assert text.index("=== 生成层指标 ===") < text.index("=== 延迟")


# ---------- 管线真的把分段耗时下发了 ----------

def test_chat_done_event_carries_stage_latency(client):
    """没有它评测就分不出检索 / 重排 / 生成 —— 分段口径靠服务端自己记。"""
    H, _uid, kb = register_and_kb(client, "lat1")
    r = client.post("/api/v1/chat/stream", headers=H,
                    json={"kb_id": kb, "question": "随便问问", "stream": True})

    done = [e for e in sse_events(r.text) if e.get("type") == "done"]
    assert done, r.text

    lat = done[-1]["latency"]
    assert set(lat) == {"retrieval_ms", "rerank_ms", "ttft_ms", "generate_ms"}
    assert lat["retrieval_ms"] is not None        # 检索器在计时
    assert lat["rerank_ms"] is not None           # 重排单独一段
    assert lat["ttft_ms"] is not None             # 假模型也会吐至少一个增量
    assert lat["generate_ms"] is not None


# ---------- 并发编排：用假 client 离线验证 ----------

_BODY = "\n".join([
    'data: {"type": "delta", "text": "甲"}',
    'data: {"type": "done", "latency": {"retrieval_ms": 10, "rerank_ms": 20,'
    ' "ttft_ms": 30, "generate_ms": 40}}',
    "data: [DONE]",
])


class _Resp:
    def __init__(self, status_code=200, body=_BODY):
        self.status_code = status_code
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        return iter(self._body.splitlines())


class _FakeClient:
    def __init__(self, resp=None):
        self.calls = 0
        self._resp = resp or _Resp()

    def stream(self, *a, **k):
        self.calls += 1
        return self._resp


def test_run_round_fires_one_request_per_question_and_collects_them():
    from evaluate_latency import _run_round

    client = _FakeClient()
    got = _run_round(client, {}, "kb", ["Q1", "Q2", "Q3"])

    assert client.calls == 3
    assert len(got) == 3
    assert all(s["status"] == 200 for s in got)
    assert all(s["retrieval"] == 10 and s["generate"] == 40 for s in got)
    assert all(s["ttft"] is not None for s in got)      # 首字由客户端自己计时


def test_one_marks_429_instead_of_faking_a_sample():
    """被并发上限挡住时不能编一条数字出来。"""
    from evaluate_latency import _one

    got = _one(_FakeClient(_Resp(status_code=429)), {}, "kb", "Q")
    assert got == {"status": 429}


def test_latency_note_states_the_stage_boundary_and_ttft_scope():
    """口径要写清「生成」算到哪为止、TTFT 含不含检索 —— 不然数字没法比。"""
    from evaluate_latency import NOTE

    assert "引用校验与落库不计入" in NOTE
    assert "含检索与重排" in NOTE
    assert "并发" in NOTE or "并发" in NOTE
