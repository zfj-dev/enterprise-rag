"""文本小工具：跨模块共用，别各写一份。"""
from __future__ import annotations

import json


def strip_list_marker(line: str) -> str:
    """去掉行首的列表标记（`- ` / `* ` / `2. ` / `3) `），**只**去标记，不吞正文。

    不能用 lstrip 把数字和点号统统削掉 —— 那会把「2024 年营收」这类以数字开头的正文削成「年营收」。
    """
    t = line.strip()
    for pre in ("-", "•", "*"):
        if t.startswith(pre):
            return t[len(pre):].strip()
    i = 0
    while i < len(t) and t[i].isdigit():
        i += 1
    if 0 < i < len(t) and t[i] in ".、)）":
        return t[i + 1:].strip()
    return t


def extract_json(text, kind: str = "{"):
    """从模型回复里抠出第一个 JSON 对象（kind="{"）或数组（kind="["）。

    模型爱把 JSON 裹在解释或 ``` 代码块里，直接 json.loads 会失败，所以按括号配平取完整片段。
    抠不出、或解析失败，返回 None。
    """
    opener, closer = (kind, "}") if kind == "{" else ("[", "]")
    s = str(text or "")
    start = s.find(opener)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == opener:
            depth += 1
        elif s[i] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


def lines_of(text) -> list[str]:
    """按行拆开，去掉行首列表标记与空行 —— 模型爱把要点写成一行一条的列表。"""
    return [t for t in (strip_list_marker(ln) for ln in str(text or "").splitlines()) if t]
