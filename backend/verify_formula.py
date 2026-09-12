"""真机验证：texify 公式识别在集成管线里是否生效。

用法:
  .venv\\Scripts\\python.exe verify_formula.py <path/to/paper.pdf>
  不带参数时自动查找 backend/uploaded_files/paper.pdf。

判据（写日志到 logs/formula-verify.log）:
  - 识别成功的公式（$$...$$ / $...$ LaTeX 块）数量 > 0
  - 残留的 <!-- formula-not-decoded --> 占位符数量理想为 0

关键：本脚本会把解析过程中 _enrich_formulas_with_ocr 的控制台输出（[formula] 诊断行）一并收进报告，
以便定位"texify 没装/模型下载失败/裁图坐标错/识别为空"到底是哪一环。
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import traceback

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

REPORT = os.path.join(BACKEND, "logs", "formula-verify.log")


def _safe(s: str) -> str:
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s.encode("utf-8", "replace").decode("utf-8")


def write_report(lines: list[str], ok: bool) -> None:
    body = "\n".join(_safe(s) for s in lines)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("=== 公式识别验证报告 ===\n")
        f.write(body + "\n")
        f.write(f"\n结果: {'PASS' if ok else 'FAIL'}\n")
    print("[doc] 报告已写入 " + REPORT)


def main() -> int:
    pdf = sys.argv[1] if len(sys.argv) > 1 else ""
    lines: list[str] = []

    def log(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    log(f"python: {sys.version.split()[0]}")
    for mod in ("docling", "transformers", "optimum", "torch", "fitz"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            log(f"{mod}: {v}")
        except Exception as e:  # noqa
            log(f"{mod}: import失败({type(e).__name__})({e})")

    # 当前公式识别开关状态
    try:
        from app.config import get_settings
        log(f"docling_formula_ocr = {get_settings().docling_formula_ocr}")
    except Exception as e:  # noqa
        log(f"读配置失败: {type(e).__name__}: {e}")

    if not pdf or not os.path.exists(pdf):
        for cand in ("paper.pdf", "毕设论文.pdf"):
            p = os.path.join(BACKEND, "uploaded_files", cand)
            if os.path.exists(p):
                pdf = p
                break
            p2 = os.path.join(BACKEND, cand)
            if os.path.exists(p2):
                pdf = p2
                break
    if not pdf or not os.path.exists(pdf):
        log("[FAIL] 找不到 PDF。请传路径参数，或把论文放 backend\\uploaded_files\\paper.pdf 后重跑。")
        write_report(lines, False)
        return 1
    log(f"PDF: {pdf}")

    text = ""
    try:
        from app.core.parser import _parse_pdf_docling

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            parsed = _parse_pdf_docling(pdf)  # 内部已含公式识别；其 print 输出被收进 buf
        text = parsed.text or ""
        log(f"解析 parser={parsed.metadata.get('parser')}  pages={parsed.page_count}  chars={len(text)}")
        console = buf.getvalue().strip()
        if console:
            log("---- [_parse_pdf_docling 控制台诊断] ----")
            for ln in console.splitlines():
                log("  " + _safe(ln))
            log("-----------------------------------")
    except Exception as e:  # noqa
        log("[FAIL] 经 _parse_pdf_docling 解析异常: " + _safe(f"{type(e).__name__}: {e}"))
        log("----- traceback -----")
        log(_safe(traceback.format_exc()))
        write_report(lines, False)
        return 1

    # 统计公式
    not_decoded = len(re.findall(r"<!-- formula-not-decoded -->", text))
    latex_blocks = re.findall(r"\$\$(.+?)\$\$", text, re.DOTALL)
    latex_inline = re.findall(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", text)
    latex_total = len(latex_blocks) + len(latex_inline)
    log(f"识别成LaTeX的公式数:   block=$$..$$ {len(latex_blocks)}  inline=$..$ {len(latex_inline)}  合计 {latex_total}")
    log(f"残留 not-decoded 占位: {not_decoded}")

    shown = 0
    for b in latex_blocks:
        if shown >= 5:
            break
        log("  [latex] " + _safe(b.strip()[:80]))
        shown += 1

    ok = latex_total > 0 and not_decoded == 0
    log("")
    log(f"结论: latex={latex_total}  not_decoded={not_decoded}")
    write_report(lines, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
