"""真实分词器（票 19 / #26）：惰性加载 + 缓存；拿不到就报「不可用」，绝不回退成估算。"""
from __future__ import annotations

import pytest

from app.core.tokenizer import setup_token_counter, try_real_token_counter


class FakeTokenizer:
    """只认字数的假分词器 —— 数得准不准不重要，重要的是「用了哪个」。"""

    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_the_real_tokenizer_is_loaded_lazily_then_cached():
    """惰性 + 缓存：装一次就够，别每来一句话就重新加载模型。"""
    loads = []

    def loader(model_id):
        loads.append(model_id)
        return FakeTokenizer()

    first, note = try_real_token_counter("Qwen/test", loader=loader)
    second, _ = try_real_token_counter("Qwen/test", loader=loader)

    assert first is second
    assert loads == ["Qwen/test"]
    assert first.label == "Qwen/test"       # 报数口径就是模型名
    assert note == ""                       # 缓存命中不算「不可用」
    assert first.count("比亚迪") == 3


def test_an_unavailable_tokenizer_is_reported_not_estimated():
    """拿不到真实分词器 -> 口径为空、原因写明；**不许**拿估算值当真实数字报。"""
    def boom(model_id):
        raise RuntimeError("下不到模型")

    counter, note = setup_token_counter("Qwen/boom", loader=boom)

    assert counter.label == ""              # 空口径 = 不许对外报 token 数字
    assert "下不到模型" in note and "Qwen/boom" in note
    assert counter.note == note             # 报告「不可用」那行直接引用它


def test_no_model_configured_means_no_download_attempt():
    """没配模型就干脆别去下载 —— 不能让一次启动卡在网络的超时重试上。"""
    calls = []

    def loader(model_id):
        calls.append(model_id)
        return FakeTokenizer()

    counter, note = try_real_token_counter("", loader=loader)

    assert calls == [] and counter is None
    assert "TOKENIZER_MODEL" in note


def test_a_real_tokenizer_reports_its_label_through_setup():
    counter, note = setup_token_counter("Qwen/test", loader=lambda m: FakeTokenizer())

    assert counter.label == "Qwen/test" and note == ""
    assert counter.count("") == 0
