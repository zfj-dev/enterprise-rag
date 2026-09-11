"""费用折算 + 价格表（票 28）：把 token 按模型单价折成**费用**，让预算与成本跨模型可比。

单价统一按「元 / 1K token」，输入与输出分开。三条硬要求：
- 价格表**可配置覆盖**内置价；
- **未知模型不按 0 算** —— 显式标注「单价未知」。按 0 算会让数字随换模型静默失真，
  而 0 元看起来又像「这次没花钱」；
- 单价**从哪儿来**也要写进口径（内置价 / 配置覆盖价），否则读的人分不清哪个是核过的。

**内置表只放 `fake`**：本地假模型不发请求，0 元是构造上的事实。真实模型的单价会随供应商
调价而失真，本项目**不内置一份无法离线核实的价目表**（同 tokenizer「拿不到就不报数」的纪律）——
要成本数字就配 `LLM_PRICE_OVERRIDES`。

token 一侧复用票 27 的记录：token 本身量不到时（`source=unavailable`）费用也无从折算。
"""
from __future__ import annotations

import json
import logging

from dataclasses import dataclass

from app.core.usage import as_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """一个模型的单价（元 / 1K token），输入与输出分开。"""

    input: float
    output: float
    origin: str = "builtin"      # builtin / override —— 单价从哪儿来，口径里要写


_BUILTIN: dict[str, Price] = {
    "fake": Price(input=0.0, output=0.0),      # 本地假模型：不发请求，真的不花钱
}


def _parse_overrides(raw) -> tuple[dict[str, Price], list[str]]:
    """解析覆盖价，返回 (价格, 问题列表)。

    **配错不阻断启动**（价格是运营配置，不该让服务起不来），但**必须留下问题清单** ——
    静默回落到内置价，会让人拿到一个看着正常的数字，却不知道自己的配置根本没生效。
    """
    if not raw:
        return {}, []
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except Exception as e:      # noqa: BLE001 —— 配置错了要说出来，但不阻断启动
            return {}, ["价格表覆盖不是合法 JSON，已整段忽略：%s" % e]

    prices: dict[str, Price] = {}
    issues: list[str] = []
    for model, value in (data or {}).items():
        try:
            prices[str(model)] = Price(input=float(value["input"]), output=float(value["output"]),
                                       origin="override")
        except Exception as e:      # noqa: BLE001 —— 单条配错只丢这一条，别连累其它
            issues.append("覆盖项 %s 格式不对（要 {input, output}），已忽略：%s" % (model, e))
    return prices, issues


class PriceTable:
    """模型 -> 单价。内置价打底，配置覆盖优先。"""

    def __init__(self, overrides: str | dict | None = None):
        self._prices = dict(_BUILTIN)
        parsed, self.warnings = _parse_overrides(overrides)
        self._prices.update(parsed)

    def price_of(self, model: str) -> Price | None:
        """该模型的单价；**没有就是 None** —— 调用方据此标注「单价未知」，别当 0。"""
        return self._prices.get(str(model or "").strip())

    def origin_of(self, model: str) -> str | None:
        """单价从哪儿来（builtin / override）；没有该模型就是 None。"""
        price = self.price_of(model)
        return price.origin if price else None


def cost_of(record: dict, table: PriceTable) -> dict:
    """给一条用量记录折算费用（元）。返回 `{cost, price_note}`。

    折不出来时 `cost` 为 **None** 并写明原因（token 量不到 / 单价未知）—— 绝不按 0 算。
    折得出来时，口径里同时写明**单价从哪来**与**token 是哪种口径**：费用数字本身要自洽可读。
    """
    in_tok, out_tok = as_int(record.get("input_tokens")), as_int(record.get("output_tokens"))
    if in_tok is None or out_tok is None:
        return {"cost": None,
                "price_note": "token 不可用，费用无从折算（%s）" % (record.get("source_note") or "")}

    model = record.get("model") or ""
    price = table.price_of(model)
    if price is None:
        return {"cost": None,
                "price_note": "单价未知：模型 %s 不在价格表里 —— 不按 0 算，"
                              "用 LLM_PRICE_OVERRIDES 补上再折" % (model or "（未标注）")}

    origin = "配置覆盖价" if price.origin == "override" else "内置价"
    caliber = {"provider": "账单口径", "local": "估算口径"}.get(record.get("source"),
                                                                record.get("source") or "口径未知")
    # 留到 1e-8 元：再粗就会把「极小但非零」的费用抹成 0，而 0 看起来像「免费」
    cost = round(in_tok / 1000 * price.input + out_tok / 1000 * price.output, 8)
    return {"cost": cost,
            "price_note": "口径：%s 单价 %g / %g 元每 1K token（输入 / 输出；%s）；token 为%s"
                          % (model, price.input, price.output, origin, caliber)}
