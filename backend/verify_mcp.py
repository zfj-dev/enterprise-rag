"""真机验证 MCP 往返：起一个**真实**的 stdio server 子进程，发现工具并调用 Calculator。

单测里传输是被 stub 的（spec 0002 要求不起真实进程）；这一个脚本是「往返真的通」的证明。

用法:  cd backend && python verify_mcp.py
报告:  logs/mcp-verify.log
"""
from __future__ import annotations

import os
import sys
import traceback

BACKEND = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BACKEND, "logs", "mcp-verify.log")

CALLS = [("(12.5-10)/10*100", 25.0),
         ("2**10", 1024),
         ("__import__('os')", None),          # 必须被拒
         ("(1).__class__", None),             # 必须被拒
         ("data[0]", None)]                   # 必须被拒


def main() -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    from app.mcp.client import StdioMCPClient

    lines = ["=== MCP 往返验证 ==="]
    client = None
    try:
        client = StdioMCPClient(sys.executable, ["-m", "app.mcp.server"], cwd=BACKEND)
        specs = client.list_tools()
        lines.append("发现工具 %d 个：" % len(specs))
        for s in specs:
            lines.append("  - %s: %s" % (s.get("name"), (s.get("description") or "")[:60]))
            lines.append("    inputSchema: %s" % s.get("inputSchema"))

        for expression, expected in CALLS:
            got = client.call_tool("Calculator", {"expression": expression})
            if expected is None:
                ok = got.get("is_error") is True and "value" not in got
                lines.append("[%s] 拒绝 %r -> %s" % ("PASS" if ok else "FAIL", expression, got))
            else:
                ok = got.get("value") == expected
                lines.append("[%s] %r = %s（期望 %s）" % ("PASS" if ok else "FAIL", expression,
                                                          got.get("value"), expected))
    except Exception:
        lines.append("fatal: " + traceback.format_exc())
    finally:
        if client is not None:
            client.close()

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
