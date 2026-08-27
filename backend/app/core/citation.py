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
