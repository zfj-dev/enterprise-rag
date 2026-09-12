# -*- coding: utf-8 -*-
"""Step4 analyze_errors.py：扫描 logs/error.log，按错误类型聚合，输出摘要，检测新错误。

用法：python analyze_errors.py [--tail N]
建议定时（每小时）：schtasks / cron 调一次。
"""
import os, re, json, sys, time
from collections import Counter, defaultdict

LOG = os.path.join(os.path.dirname(__file__), "..", "backend", "logs", "error.log")
SEEN = os.path.join(os.path.dirname(__file__), "logs", "error_seen.json")
os.makedirs(os.path.dirname(SEEN), exist_ok=True)

# 一次异常的"开头"： 时间 级别 app.error: Unhandled exception on METHOD /path
START_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.][^ ]*) \w+ app\.error: (.*)$")
# 异常类型行（最后一行 traceback 的风格）：   ClassName: message
EXC_RE = re.compile(r"^([A-Za-z_][\w\.]*)(?:Error|Exception|Interrupt|Error\b[^\n]*): (.*)$")


def parse():
    if not os.path.exists(LOG):
        return []
    groups = defaultdict(list)  # sig -> list of lines
    cur = None
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = START_RE.match(line)
            if m:
                ts, msg = m.group(1), m.group(2)
                cur = {"ts": ts, "msg": msg, "frames": []}
                groups.setdefault((msg, None), []).append(cur)
                continue
            if cur is not None:
                cur["frames"].append(line)
    return groups


def summarize():
    groups = parse()
    by_msg = Counter()
    # 进一步按 "异常类名" 聚合（从 frames 里找异常行）
    detail = defaultdict(list)
    for (msg, _), items in groups.items():
        by_msg[msg] += len(items)
        exc = None
        for frame in reversed(items[0]["frames"]):
            em = EXC_RE.match(frame.strip())
            if em:
                exc = em.group(1)
                break
        detail[(msg, exc)].append(len(items))

    print(f"=== error.log 聚合（{time.strftime('%Y-%m-%d %H:%M')}）===")
    print(f"总异常数: {sum(by_msg.values())}, 类型数: {len(detail)}")
    for (msg, exc), counts in sorted(detail.items(), key=lambda kv: -sum(kv[1]))[:20]:
        print(f"  ×{sum(counts):<4} [{'请求路径:'+msg if msg else '未知'}] 异常类={exc}")

    # 新错误检测
    current_sigs = set()
    for (msg, exc) in detail:
        current_sigs.add(f"{msg}::{exc}")
    seen = set()
    if os.path.exists(SEEN):
        try:
            seen = set(json.load(open(SEEN, encoding="utf-8")))
        except Exception:
            seen = set()
    new_sigs = current_sigs - seen
    if new_sigs:
        print(f"\n[!] 发现 {len(new_sigs)} 个新错误类型:")
        for s in sorted(new_sigs)[:15]:
            print("   -", s[:200])
    else:
        print("\n无新错误类型（与上次一致）")
    json.dump(sorted(current_sigs), open(SEEN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return len(new_sigs)


if __name__ == "__main__":
    n = summarize()
    sys.exit(1 if n else 0)
