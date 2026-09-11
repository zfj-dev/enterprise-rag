"""文本小工具：跨模块共用，别各写一份。"""
from __future__ import annotations

import json
import re


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


def truncate(text, limit: int) -> str:
    """超长就截断并加省略号 —— 看得见是被截了，别让人读成另一句。"""
    s = str(text or "")
    return s if len(s) <= limit else s[: max(1, limit - 1)] + "…"


def lines_of(text) -> list[str]:
    """按行拆开，去掉行首列表标记与空行 —— 模型爱把要点写成一行一条的列表。"""
    return [t for t in (strip_list_marker(ln) for ln in str(text or "").splitlines()) if t]


_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")


def approx_token_count(text) -> int:
    """粗略数一下 token：CJK 按字、其余按空白切词。

    **这不是真实分词器**（真实分词器见 app/core/tokenizer.py）——它只用在两处明确标注口径的地方：
    预算取舍的估算（票 17）、以及演示用假模型自报的「模拟用量」（票 30）。
    口径与 0003 里的估算计数器**保持一致**，别让同一个数在两处算出不同的值。
    """
    s = str(text or "")
    cjk = len(_CJK_RE.findall(s))
    words = len([w for w in re.split(r"\s+", _CJK_RE.sub(" ", s)) if w])
    return cjk + words
