"""RGB 中文四能力适配器（票 06）—— 把官方数据转成内部黄金集条目。

官方数据：https://github.com/chen700564/RGB （master 分支 `data/`），**只读不改**。
四条能力用哪份文件、按什么协议跑，见官方 readme：

  噪声鲁棒 noise         zh.json      给 positive + negative 文档，答案在 positive 里
  否定拒绝 rejection     zh.json      同一批问题，但**只给 negative 文档**（官方 noise_rate=1）
  信息集成 integration   zh_int.json  答案要跨多篇文档拼起来
  反事实鲁棒 counterfactual zh_fact.json 文档里混了与事实相悖的假信息

产出对得上的数字：
  噪声鲁棒 / 信息集成 / 反事实鲁棒 → 期望事实命中率（accuracy）
  否定拒绝                        → 拒答率（复用票 03 的拒答判据）

**数据不在本地就明说**（DatasetMissing），绝不产出看着正常的假数字。

**官方文件的真实形状（#52，真机踩过）**：
- 它们叫 `.json`，但内容是 **JSONL —— 一行一条**。整份当单个 JSON 文档解析会 `Extra data`。
  读入两种都认（单个数组 / JSONL），坏行报出**行号**。
- 字段形状不统一：`answer` 是**列表**（`["12名"]`），`zh_int.json` 的 `positive` 还是
  **list 套 list**。一律摊平处理 —— `str(["12名"])` 会得到字面量 `"['12名']"`，
  跟答案永远匹配不上，而报告只显示「命中率 0」，看不出是解析坏了。

**已知边界**：官方的 `answer` 常是多值（如 `["1月3日", "1月12日"]`）—— 实测
`zh_int.json` **100/100 条**都是多值、`zh.json` 44/300。评测核心的 `expect` 是**单个字符串**，
这里取摊平后的**第一项**，即**只核第一个值**，属于**偏宽松**的口径 —— 信息集成那一组基本等于
「只核了官方答案的第一项」，看它的数字时要知道这一点（RGB 报告里也印了这句）。
"""
from __future__ import annotations

import json
import os

# 能力 key -> 这一能力的全部定义。**一张表**：加一种能力只改这里一处。
#   file      官方文件名（一个文件可能服务多种能力）
#   label     报告里显示的名字
#   doc_keys  该把条目的哪些文档喂进去（这就是各能力协议上的差别）
ABILITIES = {
    "noise": {"file": "zh.json", "label": "噪声鲁棒",
              "doc_keys": ("positive", "negative")},
    "rejection": {"file": "zh.json", "label": "否定拒绝",
                  "doc_keys": ("negative",)},             # 只给噪声文档 —— 官方拒绝协议
    "integration": {"file": "zh_int.json", "label": "信息集成",
                    "doc_keys": ("positive", "negative")},
    "counterfactual": {"file": "zh_fact.json", "label": "反事实鲁棒",
                       "doc_keys": ("positive", "positive_wrong")},   # 正确的 + 与事实相悖的
}


def ability_label(ability: str) -> str:
    """能力 key -> 报告里显示的名字。"""
    return ABILITIES[ability]["label"]


class DatasetMissing(RuntimeError):
    """官方数据不在本地。必须显式报错 —— 不能拿空结果或旧数字糊弄过去。"""


def dataset_path(data_dir: str, ability: str) -> str:
    """这份能力对应的官方文件路径。"""
    if ability not in ABILITIES:
        raise KeyError("未知能力：%s（可选：%s）" % (ability, " / ".join(ABILITIES)))
    return os.path.join(data_dir, ABILITIES[ability]["file"])


def available(data_dir: str) -> dict:
    """四种能力各自「数据在不在」—— 报告里据此说明哪些跑不了。"""
    return {ability: os.path.isfile(dataset_path(data_dir, ability)) for ability in ABILITIES}


def load_entries(data_dir: str, ability: str) -> list:
    """读官方数据 → 内部黄金集条目（含这一能力该喂哪些文档）。

    条目形如 {"question", "expect"?, "negative"?, "group", "documents": [...]}，
    `documents` 是本条要建索引的官方文档原文 —— 官方数据本身一个字节都不改。
    """
    path = dataset_path(data_dir, ability)
    if not os.path.isfile(path):
        raise DatasetMissing(
            "缺少 RGB 官方数据 %s（能力：%s）。请从 https://github.com/chen700564/RGB 的 "
            "data/ 目录取到 %s 后再跑 —— 这里不会用假数据顶替。" % (path, ability, data_dir))

    raw = _read_items(path)
    keys = ABILITIES[ability]["doc_keys"]
    entries: list = []
    for item in raw:
        question = str(item.get("query", "")).strip()
        if not question:
            continue
        entry = {
            "question": question,
            "group": ability_label(ability),
            "documents": _documents(item, keys),
        }
        if ability == "rejection":
            entry["negative"] = True                    # 文档里就没有答案，期望拒答
        else:
            # 取摊平后的第一项：官方 answer 是列表，`str()` 会变成 "['12名']" 匹配不上（#52）。
            # 多值条目这里只核第一个 —— 偏宽松的口径，模块文档里写明了。
            answers = _flat_texts(item.get("answer"))
            entry["expect"] = answers[0] if answers else ""
        entries.append(entry)
    return entries


def _read_items(path: str) -> list:
    """读官方条目 —— 官方文件是 **JSONL（一行一条）**。

    仓库里它们叫 `.json`，但整份当单个 JSON 文档解析会报 `Extra data`：里面是一行一条。
    两种都认（先整文档、再逐行），都读不出才算数据不可用。
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise DatasetMissing("%s 读不出来（%s）：%s" % (path, type(e).__name__, e))

    whole_error = None
    try:
        raw = json.loads(text)
    except ValueError as e:
        raw, whole_error = None, e
    if isinstance(raw, list):
        return raw                       # 单个数组的版本（旧导出 / 测试夹具）

    items: list = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except ValueError as e:
            # 首行就解析不了 → 这文件多半**根本不是** JSONL（比如缩进过的单个 JSON 对象）。
            # 那时只报「第 1 行不是合法 JSON」是**指错了地方**（那行往往就是个 `[` / `{`），
            # 得把「整份当单文档也没解析成功」这个真正的原因一并说出来。
            extra = ""
            if not items and whole_error is not None:
                extra = "；整份当单个 JSON 也没解析成功（%s）—— 上游导出格式可能不是你预期的那种" % whole_error
            # 其余情况报出**行号** —— 几 MB 的文件里没有行号没法找
            raise DatasetMissing("%s 第 %d 行不是合法 JSON（%s）%s" % (path, lineno, e, extra))
    if not items:
        raise DatasetMissing("%s 里没有任何条目（既不是 JSON 数组，也不是 JSONL）" % path)
    return items


def _flat_texts(value) -> list:
    """把官方字段摊平成字符串列表 —— `str` / `list[str]` / **`list[list[str]]`** 都认。

    **不能 `str(value)` 了事**：`["12名"]` 会变成字面量 `"['12名']"`，跟答案文本永远匹配不上，
    而报告只会显示「命中率 0」—— 看上去像效果差，看不出是解析坏了（#52）。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list = []
        for v in value:
            out.extend(_flat_texts(v))
        return out
    if isinstance(value, (dict, set)):
        # 落到这里说明字段形状又是新的。`str(dict)` 会得到一坨 repr —— 拿它去匹配就是在编事实。
        # 宁可报出来：#52 的教训就是「形状假设错了会静默产出 0 分，看不出是解析坏了」。
        raise DatasetMissing("官方字段出现了本适配器不认识的形状（%s）—— 别把它的 repr 当事实去匹配"
                             % type(value).__name__)
    text = str(value).strip()            # 数字 / 布尔等标量：转字符串是合理的
    return [text] if text else []


def _documents(item: dict, keys) -> list:
    """按能力取官方文档原文（摊平后），跳过空串。"""
    out: list = []
    for key in keys:
        out.extend(_flat_texts(item.get(key)))
    return out
