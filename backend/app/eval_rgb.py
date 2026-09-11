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

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:      # 读不了 / JSON 坏了，都算「数据不可用」，要明说
        raise DatasetMissing("%s 读不出来（%s）：%s" % (path, type(e).__name__, e))
    if not isinstance(raw, list):
        raise DatasetMissing("%s 不是 RGB 的条目数组（顶层应是 list）" % path)

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
            entry["expect"] = str(item.get("answer", ""))
        entries.append(entry)
    return entries


def _documents(item: dict, keys) -> list:
    """按能力取官方文档原文，跳过空串。"""
    out: list = []
    for key in keys:
        for doc in item.get(key) or []:
            text = str(doc).strip()
            if text:
                out.append(text)
    return out
