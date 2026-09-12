"""Generate deterministic sample docs (docx/xlsx/md) for RAG golden-set evaluation.
Run:  python scripts/gen_sample_docs.py
Outputs to backend/data/samples/. ASCII comments only (PowerShell GBK caution).
"""
import io
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "samples")
os.makedirs(BASE, exist_ok=True)


def make_docx(path: str) -> None:
    from docx import Document
    from docx.shared import Pt

    d = Document()
    d.add_heading("员工手册", level=0)
    d.add_paragraph("员工入职后享有每年7天带薪年假。")
    d.add_paragraph("年假申请需提前3天在OA系统提交,由直属主管审批。")
    d.add_paragraph("病假需提供医院证明,3天以上(不含3天)由部门总监审批。")
    d.add_paragraph("公司实行弹性工作制,核心工作时间为10:00-16:00。")
    # table: 假期类型 | 时长 | 审批人
    t = d.add_table(rows=1, cols=3)
    hdr = t.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "假期类型", "时长", "审批人"
    for row in (("年假", "7天", "直属主管"), ("病假", "按病假条", "部门总监"), ("事假", "每年5天", "直属主管")):
        c = t.add_row().cells
        c[0].text, c[1].text, c[2].text = row
    d.save(path)


def make_xlsx(path: str) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "2025"
    ws.append(["区域", "Q1营收(万元)"])
    for row in (("华东", 560), ("华南", 430), ("华北", 210)):
        ws.append(list(row))
    ws2 = wb.create_sheet("汇总")
    ws2.append(["指标", "数值"])
    ws2.append(["2025年一季度营业总收入(万元)", 1200])
    wb.save(path)


def make_md(path: str) -> None:
    content = (
        "# 产品说明\n\n"
        "本系统支持PDF、Word、Excel批量导入。\n\n"
        "单次最多导入1000个文件。\n\n"
        "支持语义检索与关键词检索两种方式。\n\n"
        "内置RBAC权限控制,支持admin/uploader/viewer三种角色。\n\n"
        "问答结果提供来源引用,并标注页码。\n"
    )
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    make_docx(os.path.join(BASE, "员工手册.docx"))
    make_xlsx(os.path.join(BASE, "销售数据.xlsx"))
    make_md(os.path.join(BASE, "产品说明.md"))
    print("generated samples in", BASE)
