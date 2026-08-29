"""诊断：docling 解析 paper.pdf 后，表3.1 的正文在不在、chunk 怎么切的。
用法: .venv\Scripts\python.exe diagnose_table.py   报告 logs/table-diagnose.log
"""
from __future__ import annotations
import os, sys
BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

REPORT = os.path.join(BACKEND, "logs", "table-diagnose.log")
_lines: list[str] = []
def log(s: str = "") -> None:
    _lines.append(str(s)); print(s)

def main() -> int:
    pdf = os.path.join(BACKEND, "paper.pdf")
    if not os.path.exists(pdf):
        log("未找到 paper.pdf"); _write(); return 1

    from app.core.parser import ParserRouter
    parsed = ParserRouter().parse(pdf, "paper.pdf")
    log(f"parser={parsed.metadata.get('parser')}  page_count={parsed.page_count}  chars={len(parsed.text)}")
    text = parsed.text

    # 1) 表3.1 正文关键词在 docling markdown 里吗？
    log("\n===== 关键词扫描 =====")
    found = False
    for kw in ["RTX", "代码编译器", "PyCharm", "操作系统", "CUDA"]:
        idx = text.find(kw)
        log(f"  '{kw}': {'命中 @'+str(idx) if idx>=0 else '未找到'}")
        if idx >= 0 and not found:
            log(f"\n===== '{kw}' 附近 800 字符 =====")
            log(text[max(0,idx-300):idx+500])
            found = True

    # 2) 表3.1 标题附近 (caption 与表格的相对位置)
    for pat in ["表 3 . 1", "表3.1", "实验环境配置"]:
        idx = text.find(pat)
        if idx >= 0:
            log(f"\n===== 标题 '{pat}' @{idx} 附近 1200 字符 =====")
            log(text[max(0,idx-200):idx+1000])
            break

    # 3) chunk 切分: 哪些 chunk 含表格关键词
    from app.core.chunker import ParentChildChunker
    chunker = ParentChildChunker()
    units = chunker.chunk(text, doc_id="diag", page_num=1)
    log(f"\n===== chunk 总数 {len(units)} =====")
    kws = ("表 3 . 1", "表3.1", "RTX", "代码编译器", "PyCharm", "实验环境配置")
    hits = [c for c in units if any(k in c["content"] for k in kws)]
    log(f"含表格关键词的 chunk: {len(hits)} 个")
    for c in hits[:12]:
        log(f"\n[{c['id']}] type={c['chunk_type']}")
        log(c["content"][:250])
    _write()
    return 0

def _write() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[doc] 报告已写入 {REPORT}")

if __name__ == "__main__":
    sys.exit(main())
