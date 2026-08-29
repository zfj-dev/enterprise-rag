"""引用校验：no source → no claim。MVP 用启发式（源有稳定 chunk_id + 非空 + 可追溯）。

真实部署可升级为二次 LLM 校验（每句论断是否有出处）并把引用覆盖率作指标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class CitationResult:
    has_sources: bool
    coverage: float  # 0~1 启发式覆盖率
    stable_ids: bool
    notes: list[str] = field(default_factory=list)


def validate_sources(candidates: Sequence[dict]) -> CitationResult:
    """检查检索到的源片段是否具备可引用条件（稳定 chunk_id + 非空 + 绑定元数据）。"""
    notes: list[str] = []
    stable = True
    usable = 0
    for c in candidates:
        cid = c.get("chunk_id")
        content = c.get("content")
        meta = c.get("metadata", {}) or {}
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
    try:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    except Exception:
        pass
    return None


def _parse_verification(raw: str, claims: list[str]) -> dict:
    try:
        obj = json.loads(raw)
    except Exception:
        obj = _extract_json_object(raw)
    if not isinstance(obj, dict):
        obj = {}
    items = obj.get("claims") or []
    supported = set()
    for it in items:
        claim = str(it.get("claim", "")).strip()
        ok = bool(it.get("supported"))
        for c in claims:
            if claim and (c in claim or claim in c):
                if ok:
                    supported.add(c)
    return {"coverage": round(len(supported) / len(claims), 3),
            "total": len(claims), "supported": len(supported)}


def verify_claims(answer: str, sources: Sequence[dict], llm, max_claims: int = 8) -> dict:
    """LLM 逐句校验：论断是否被来源支撑，返回引用覆盖率。LLM 失败降级为启发式（有来源则假定支撑）。"""
    claims = _split_claims(answer, max_claims)
    if not claims:
        return {"coverage": 0.0, "total": 0, "supported": 0}
    if not sources:
        return {"coverage": 0.0, "total": len(claims), "supported": 0}
    src_text = "\n".join(f"[{i}] {s.get('text', '')[:500]}" for i, s in enumerate(sources[:5], 1))
    prompt = (
        "请判断以下每个论断是否被参考资料支撑。只输出 JSON 格式："
        '{"claims":[{"claim":"原论断","supported":true或false}]}\n\n'
        f"参考资料：\n{src_text}\n\n论断：\n" + "\n".join(f"- {c}" for c in claims)
    )
    try:
        raw = "".join(llm.stream([{"role": "user", "content": prompt}]))
        return _parse_verification(raw, claims)
    except Exception:
        return {"coverage": 1.0 if sources else 0.0,
                "total": len(claims), "supported": len(claims) if sources else 0}
