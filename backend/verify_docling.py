"""真机验证：Docling 能否解析指定 PDF 并提取表格。

用法:
  .venv\\Scripts\\python.exe verify_docling.py <path/to/paper.pdf>
  不带参数时自动查找 backend/paper.pdf 或 backend/verify_paper.pdf。
  （建议：把论文 PDF 复制成 backend\\paper.pdf，然后直接跑 scripts\\verify_docling.ps1）

写报告到 logs/docling-verify.log。判据:
  - 解析器为 docling（未回退 pymupdf）
  - 提取到 Markdown 管道表格行（| ... |）或 <table>
  - 命中表/图题（如 表3.1 / Table 3.1）

加固点（2026-08-29）:
  - 异常完整堆栈强制落盘（含 docling/onnxruntime/python 版本诊断）
  - 失败时用 PyMuPDF 对照组确认 PDF 本身可读
"""
from __future__ import annotations

import os
import re
import sys
import traceback

os.environ["HF_HUB_OFFLINE"] = "0"  # 允许联网：首次需下载 docling 版面模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内走镜像
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Windows 无开发者模式时禁止符号链接，改复制
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 关 Xet(CAS) 大文件后端，回落普通 HTTP 走 hf-mirror

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

REPORT = os.path.join(BACKEND, "logs", "docling-verify.log")


def _safe(s: str) -> str:
    """sanitize to bytes-writable utf-8 (drop surrogates / undecodable)."""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s.encode("utf-8", "replace").decode("utf-8")


def write_report(lines: list[str], ok: bool) -> None:
    body = "\n".join(_safe(s) for s in lines)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("=== Docling 论文 PDF 解析验证报告 ===\n")
        f.write(body + "\n")
        f.write(f"\n结果: {'PASS' if ok else 'FAIL'}\n")
    print("[doc] 报告已写入 " + REPORT)


def main() -> int:
    pdf = sys.argv[1] if len(sys.argv) > 1 else ""
    lines: list[str] = []

    def log(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    # ---- 版本诊断 ----
    log(f"python: {sys.version.split()[0]}")
    for mod in ("docling", "docling_parse", "pypdfium2", "onnxruntime", "transformers", "torch", "fitz"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            log(f"{mod}: {v}")
        except Exception as e:  # noqa
            log(f"{mod}: import失败({type(e).__name__})")

    if not pdf or not os.path.exists(pdf):
        for cand in ("paper.pdf", "verify_paper.pdf"):
            p = os.path.join(BACKEND, cand)
            if os.path.exists(p):
                pdf = p
                break
    if not pdf or not os.path.exists(pdf):
        log("[FAIL] 找不到 PDF。请把论文复制为 backend\\paper.pdf 后重跑，或传路径参数。")
        write_report(lines, False)
        return 1

    log(f"PDF: {pdf}")
    log(f"大小: {os.path.getsize(pdf) / 1024:.0f} KB")

    # ---- 1) 直接调 docling 解析 ----
    parser = ""
    text = ""
    try:
        from app.core.parser import _parse_pdf_docling

        parsed = _parse_pdf_docling(pdf)
        parser = parsed.metadata.get("parser", "?")
        text = parsed.text or ""
        log(f"[{'PASS' if parser == 'docling' else 'FAIL'}] 直接 docling 解析  "
            f"parser={parser}  pages={parsed.page_count}  chars={len(text)}")
    except Exception as e:  # noqa
        log("[FAIL] docling 解析异常: " + _safe(f"{type(e).__name__}: {e}"))
        log("----- 完整 traceback -----")
        log(_safe(traceback.format_exc()))
        # ---- 对照组：PyMuPDF 确认 PDF 可读 ----
        try:
            from app.core.parser import _parse_pdf
            t, cnt, pages = _parse_pdf(pdf)
            log(f"[对照] PyMuPDF 可解析该 PDF: pages={cnt} chars={len(t)}  (说明 PDF 本身没坏，问题在 docling)")
            # 对照表题
            caps_ctl = re.findall(r"[表图]\s*[0-9]+\s*[-–.]?\s*[0-9]*", t)
            log(f"[对照] PyMuPDF 提取到表/图题: {list(dict.fromkeys(caps_ctl))[:10]}")
        except Exception as e2:  # noqa
            log("[对照] PyMuPDF 也失败: " + _safe(f"{type(e2).__name__}: {e2}"))
        write_report(lines, False)
        return 1

    # ---- 2) 走 ParserRouter 确认不会回退 PyMuPDF ----
    ok_router = False
    try:
        from app.core.parser import ParserRouter

        rt = ParserRouter().parse(pdf, os.path.basename(pdf))
        rp = rt.metadata.get("parser", "")
        ok_router = rp == "docling"
        log(f"[{'PASS' if ok_router else 'FAIL'}] 路由解析  parser={rp or 'pymupdf(回退)'}（期望 docling）")
    except Exception as e:  # noqa
        log("[FAIL] 路由解析异常: " + _safe(f"{type(e).__name__}: {e}"))

    # ---- 3) 表格行 ----
    rows = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
    html_table = "<table" in text.lower()
    has_table = bool(rows) or html_table
    log(f"[{'PASS' if has_table else 'FAIL'}] 表格提取  pipe_rows={len(rows)}  html_table={html_table}")

    # ---- 4) 表/图题 ----
    caps = re.findall(r"[表图]\s*[0-9]+\s*[-–.]?\s*[0-9]*", text)
    caps += re.findall(r"Table\s+[0-9]+", text, re.IGNORECASE)
    caps = list(dict.fromkeys(caps))[:15]
    log(f"[{'PASS' if caps else 'FAIL'}] 表/图题命中: {caps if caps else '未找到'}")

    # ---- 5) 最长连续表格块预览 ----
    if rows:
        blocks: list[list[str]] = []
        cur: list[str] = []
        for ln in text.splitlines():
            if ln.lstrip().startswith("|"):
                cur.append(ln)
            else:
                if cur:
                    blocks.append(cur)
                    cur = []
        if cur:
            blocks.append(cur)
        best = max(blocks, key=len)
        log("")
        log("--- 最长表格块预览(前 12 行) ---")
        for ln in best[:12]:
            log("  " + _safe(ln[:120]))

    ok = parser == "docling" and ok_router and has_table and bool(caps)
    log("")
    log(f"结论: parser={parser} 表格行={len(rows)} 表题={len(caps)}")
    write_report(lines, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
