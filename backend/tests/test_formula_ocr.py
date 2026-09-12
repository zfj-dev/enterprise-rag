"""公式 OCR（docling 标出的公式区域 → pix2tex LaTeX）的集成回归测试。

这类测试不跑真实 docling/pix2tex（它们需模型与 GPU），只保证控制流安全：
关闭开关时透传、异常时优雅跳过，不触发模型下载、不崩主流程。
"""
from app.config import get_settings
from app.core.parser import _enrich_formulas_with_ocr


def test_formula_ocr_config_default_true():
    """开关默认开（用户已选择集成 pix2tex）；若哪天想彻底关，改这里。"""
    assert get_settings().docling_formula_ocr is True


def test_enrich_skips_when_disabled(monkeypatch):
    """关闭开关时应直接透传 res，不触碰 docling/pix2tex。"""
    monkeypatch.setattr(get_settings(), "docling_formula_ocr", False)
    res = _enrich_formulas_with_ocr("sentinel", "x.pdf")
    assert res == "sentinel"


def test_enrich_graceful_when_no_document():
    """开启开关但传入的 res 无 document（docling 不可用/异常）→ 不崩溃，返回原 res。"""
    res = _enrich_formulas_with_ocr(None, "x.pdf")
    assert res is None
