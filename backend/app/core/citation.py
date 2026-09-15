"""引用校验：no source → no claim。MVP 用启发式（源有稳定 chunk_id + 非空 + 可追溯）。

真实部署可升级为二次 LLM 校验（每句论断是否有出处）并把引用覆盖率作指标。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.utils.text import extract_json

logger = logging.getLogger(__name__)


@dataclass
class CitationResult:
    has_sources: bool
    coverage: float  # 0~1 启发式覆盖率
    stable_ids: bool
    notes: list[str] = field(default_factory=list)


def validate_sources(candidates: Sequence[dict]) -> CitationResult:
    """检查源片段是否具备可引用条件（稳定 chunk_id + 非空）。

    两种形状都认：候选块的 content / 对外 sources 的 text。
    """
    notes: list[str] = []
    stable = True
    usable = 0
    for c in candidates:
        cid = c.get("chunk_id")
        # 候选块叫 content，对外 sources 叫 text —— 同一个东西的两种叫法，都得认
        content = c.get("content") or c.get("text")
        if not cid:
            stable = False
            notes.append("存在无稳定 chunk_id 的源")
        if content and cid:
            usable += 1
    coverage = min(1.0, usable / max(len(candidates), 1))
    if usable == 0:
        notes.append("无可用引用源")
    return CitationResult(has_sources=usable > 0, coverage=round(coverage, 3),
                          stable_ids=stable, notes=notes)


def apply_no_source_no_claim(answer: str, result: CitationResult) -> str:
    """若没有任何可用引用，则不允许凭模型知识作答，改为明确"无法确定"。"""
    if not result.has_sources:
        return "根据现有资料无法确定（未检索到可引用的内容）。"
    return answer


import json
import re


def _split_claims(answer: str, max_claims: int = 8) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\n])", answer or "")
    return [p.strip() for p in parts if p.strip()][:max_claims]


def _extract_json_object(text: str):
    obj = extract_json(text, "{")      # 抽取口径与评测裁判共用一份（app/utils/text.py）
    if obj is None:
        logger.warning("提取 JSON 对象失败")
    return obj


def _parse_verification(raw: str, claims: list[str]):
    """解析裁判回复；**解析不了返回 None**。

    「判不了」与「判成 0 条支撑」不是一回事：前者是未知，后者是结论。
    以前这里解析失败会回落成 coverage=0.0 —— 那等于说「一句依据都没有」，
    真值是「压根没判成」。下游据此拒答就是拿未知当结论。
    """
    try:
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("解析验证 JSON 失败,回退提取: %s", e)
        obj = _extract_json_object(raw)
    if not isinstance(obj, dict) or not obj.get("claims"):
        return None
    items = obj.get("claims") or []
    supported = set()
    matched = 0
    for it in items:
        claim = str(it.get("claim", "")).strip()
        ok = bool(it.get("supported"))
        for c in claims:
            if claim and (c in claim or claim in c):
                matched += 1
                if ok:
                    supported.add(c)
    # 裁判回了论断、却一条都对不上我们发的那些 -> 是**对不上**，不是「一条都没被支撑」。
    # 这条以前只影响一个指标；现在 coverage 决定要不要拒答，认错方向就会把好答案拦掉。
    if items and not matched:
        logger.warning("引用校验：裁判回的论断与发出去的一条都对不上，这次按「没校验成」处理")
        return None
    return {"coverage": round(len(supported) / len(claims), 3),
            "total": len(claims), "supported": len(supported)}


def verify_claims(answer: str, sources: Sequence[dict], llm, max_claims: int = 8) -> dict:
    """LLM 逐句校验：论断是否被来源支撑。返回 {"coverage", "total", "supported", "verified"}。

    **「没校验成」不许写成数字**：LLM 调不通或回复解析不了时 `verified=False`、
    `coverage=None`。以前这里回落成 1.0（"有来源就假定支撑"）—— 那是在编一个
    100%，上游会据此以为有依据，评测也会把这个假的 100% 平均进覆盖率。
    「没论断」或「没来源」是**确定结论**（支撑数就是 0），仍给 0.0 且 `verified=True`。
    """
    claims = _split_claims(answer, max_claims)
    if not claims or not sources:
        return {"coverage": 0.0, "total": len(claims), "supported": 0, "verified": True}
    src_text = "\n".join(f"[{i}] {s.get('text', '')[:500]}" for i, s in enumerate(sources[:5], 1))
    prompt = (
        "请判断以下每个论断是否被参考资料支撑。只输出 JSON 格式："
        '{"claims":[{"claim":"原论断","supported":true或false}]}\n\n'
        f"参考资料：\n{src_text}\n\n论断：\n" + "\n".join(f"- {c}" for c in claims)
    )
    try:
        raw = "".join(llm.stream([{"role": "user", "content": prompt}]))
    except Exception as e:
        logger.warning("引用校验 LLM 调用失败：这次没有覆盖率（不是 0，也不是 1）: %s", e)
        return {"coverage": None, "total": len(claims), "supported": None, "verified": False,
                "note": "引用校验失败：%s" % e}
    parsed = _parse_verification(raw, claims)
    if parsed is None:
        return {"coverage": None, "total": len(claims), "supported": None, "verified": False,
                "note": "引用校验回复解析不了"}
    parsed["verified"] = True
    return parsed


# 覆盖率不足以支撑回答时的兜底话术。**写清是哪一种「答不了」** ——
# 与 apply_no_source_no_claim 的「未检索到可引用的内容」不是一回事：这里检索到了东西，
# 只是回答里的论断在里头找不到依据。
UNSUPPORTED_ANSWER = "根据现有资料无法确定（回答中的论断未能在参考资料中找到依据）。"


def coverage_too_low(verification: dict, min_coverage: float) -> bool:
    """覆盖率是否**不高于**门槛 —— 与 `apply_coverage_guard` 共用这一处判据。

    调用方要记「这次拦了没有」时问这个，别拿返回值跟常量比字符串。
    """
    if not verification.get("verified"):
        return False            # 没校验成 = 未知，不拦
    cov = verification.get("coverage")
    return cov is not None and cov <= min_coverage


def apply_coverage_guard(answer: str, verification: dict, min_coverage: float) -> str:
    """覆盖率不高于门槛 -> 不拿模型自己的知识作答，改成明确「无法确定」（票 B）。

    **只在 `verified` 为真时才拦**：「这次没校验成」是**未知**，不是「没有支撑」——
    拿未知去否掉一个本来正确的回答，跟「有依据却硬答」一样是假结论。
    门槛语义是「必须**严格大于**」：默认 0.0 = 一条依据都没有就拒答。
    """
    if not coverage_too_low(verification, min_coverage):
        return answer
    logger.info("引用覆盖率 %s 未超过门槛 %s，改为拒答", verification.get("coverage"), min_coverage)
    return UNSUPPORTED_ANSWER
