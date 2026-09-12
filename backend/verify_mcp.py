"""真机验证 MCP 往返：起一个**真实**的 stdio server 子进程，发现三个工具并调用它们。

单测里传输是被 stub 的（spec 0002 要求不起真实进程）；这一个脚本是「外部客户端真能挂载」
的证明：先是 Calculator（不依赖上下文），再是**带身份挂载**的 KbRetrieve / SqlQuery。

用法:  cd backend && python verify_mcp.py        （或 scripts/verify_mcp.ps1）
报告:  logs/mcp-verify.log（退出码 0 = 全 PASS）
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BACKEND, "logs", "mcp-verify.log")

DOC_TEXT = "比亚迪2025年营业收入为803.96亿元。公司还提到了火星基地计划。"
USERNAME = "__mcp_verify__"

CALC_CALLS = [("(12.5-10)/10*100", 25.0),
              ("2**10", 1024),
              ("__import__('os')", None),          # 必须被拒
              ("(1).__class__", None),             # 必须被拒
              ("data[0]", None)]                   # 必须被拒


def _seed() -> str:
    """建一个带文档的库，供「带身份挂载」那半边用。返回 kb_id。"""
    from app.core.container import build_runtime
    from app.db.session import SessionLocal
    from app.eval_setup import ensure_schema, eval_user, ingest_file, new_kb

    path = os.path.join(tempfile.mkdtemp(prefix="mcp_verify_"), "annual.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(DOC_TEXT)

    ensure_schema()
    db = SessionLocal()
    try:
        user = eval_user(db, USERNAME)
        kb = new_kb(db, user.id, "MCP 验证库")
        ingest_file(db, build_runtime(), path, user.id, kb.id)
        return kb.id
    finally:
        db.close()


def _safe(s: str) -> str:
    """落盘前清掉非法代理字符 —— 子进程输出里混进坏字节时，报告不能跟着写失败。"""
    return str(s).encode("utf-8", "replace").decode("utf-8")


def _report(lines: list[str]) -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_safe(s) for s in lines) + "\n")
    print(REPORT)
    print("\n".join(_safe(s) for s in lines[:6]))


def main() -> int:
    from app.mcp.client import StdioMCPClient

    lines = ["=== MCP 往返验证（外部客户端挂载）==="]
    checks: list[bool] = []
    kb_id = None
    client = None
    try:
        kb_id = _seed()
        lines.append("已备好一个带文档的库：kb=%s user=%s" % (kb_id, USERNAME))

        # 带身份挂载：外部客户端起子进程时把身份与范围显式传进去（客户端传参一律不看）。
        # env 必须显式给：SDK 只把一份白名单环境交给子进程，DATABASE_URL 不在里面。
        client = StdioMCPClient(sys.executable,
                                ["-m", "app.mcp.server", "--user", USERNAME, "--kb", kb_id],
                                cwd=BACKEND, env=dict(os.environ))
        specs = client.list_tools()
        names = sorted(s.get("name") for s in specs)
        checks.append(names == ["Calculator", "KbRetrieve", "SqlQuery"])
        lines.append("[%s] 发现工具 %d 个：%s" % ("PASS" if checks[-1] else "FAIL",
                                                len(specs), ", ".join(names)))
        for s in specs:
            schema = s.get("inputSchema") or {}
            ok = bool((s.get("description") or "").strip()) and schema.get("type") == "object"
            checks.append(ok)
            lines.append("[%s] %s 有描述与 object schema：%s"
                         % ("PASS" if ok else "FAIL", s.get("name"), schema))

        for expression, expected in CALC_CALLS:
            got = client.call_tool("Calculator", {"expression": expression})
            if expected is None:
                ok = got.get("is_error") is True and "value" not in got
                lines.append("[%s] 拒绝 %r -> %s" % ("PASS" if ok else "FAIL", expression, got))
            else:
                ok = got.get("value") == expected
                lines.append("[%s] %r = %s（期望 %s）" % ("PASS" if ok else "FAIL", expression,
                                                          got.get("value"), expected))
            checks.append(ok)

        # 带上下文的工具：经真实 stdio 往返调用，范围由挂载配置注入
        ret = client.call_tool("KbRetrieve", {"query": "营业收入"})
        sources = ret.get("sources") or []
        ok = bool(sources) and any("803.96" in str(s.get("text", "")) for s in sources)
        checks.append(ok)
        lines.append("[%s] KbRetrieve 返回 %d 段来源，命中期望事实：%s"
                     % ("PASS" if ok else "FAIL", len(sources),
                        sources[0] if sources else ret))

        sql = client.call_tool("SqlQuery", {"table": "documents", "fields": ["filename", "status"]})
        rows = sql.get("rows") or []
        ok = sql.get("count", 0) >= 1 and any("annual" in str(r.get("filename", "")) for r in rows)
        checks.append(ok)
        lines.append("[%s] SqlQuery 查到 %d 行文档元数据：%s"
                     % ("PASS" if ok else "FAIL", len(rows), rows[:3]))

        bad = client.call_tool("SqlQuery", {"table": "users"})     # 白名单外的表必须被拒
        ok = bad.get("is_error") is True
        checks.append(ok)
        lines.append("[%s] 白名单外的表被拒：%s" % ("PASS" if ok else "FAIL", bad))
    except Exception:
        lines.append("fatal: " + traceback.format_exc())
        checks.append(False)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as e:      # noqa: BLE001 —— 关进程出错不该拦着落报告
                lines.append("关闭客户端出错（不影响结论）：%s" % e)
        if kb_id:
            try:
                from app.db.session import SessionLocal
                from app.eval_setup import drop_kb

                db = SessionLocal()
                drop_kb(db, kb_id)
                db.close()
            except Exception as e:      # noqa: BLE001 —— 清理失败不该毁掉验证结论
                lines.append("清库失败（不影响结论）：%s" % e)

    passed = sum(1 for c in checks if c)
    lines.append("")
    lines.append("结果: %d/%d PASS" % (passed, len(checks)))
    _report(lines)
    return 0 if checks and passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
