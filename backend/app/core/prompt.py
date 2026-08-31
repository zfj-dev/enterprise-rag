"""提示词构建（含引用规则、no source → no claim、敏感信息拒答）。"""
from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = (
    "你是企业知识库助手。请基于以下检索到的参考资料回答问题。\n"
    "规则：\n"
    "1. 如果参考资料足以回答，请直接回答，并在关键论断后标注 [来源: 文档名, 第X页]。\n"
    "2. 如果参考资料不足，请明确说\"根据现有资料无法确定\"，绝不编造。\n"
    "3. 涉及密码、薪资、个人隐私等敏感信息，拒绝回答。\n"
    "4. 回答语言与用户提问语言一致。"
)


def build_prompt(
    query: str,
    context: str,
    history: Sequence[dict] | None = None,
    graph_context: str | None = None,
    system: str = SYSTEM_PROMPT,
    enum_hint: str | None = None,
) -> str:
    parts = [system]
    if context:
        parts.append(f"【参考资料】\n{context}")
    if graph_context:
        parts.append(f"【知识图谱关联】\n{graph_context}")
    if history:
        lines = []
        for h in history[-3:]:
            lines.append(f"用户：{h.get('user', '')}\n助手：{h.get('assistant', '')}")
        parts.append("【历史对话】\n" + "\n".join(lines))
    if enum_hint:
        parts.append(f"【注意】{enum_hint}")
    parts.append(f"【用户问题】\n{query}")
    return "\n\n".join(parts)


def build_rewrite_prompt(user_question: str, history: Sequence[dict] | None = None) -> str:
    hist = history or []
    ctx = "\n".join(f"用户：{h.get('user','')} -> 助手：{h.get('assistant','')}" for h in hist[-2:])
    return (
        "你是查询改写助手。结合最近对话，把用户当前问题改写成能独立检索、指代清晰的问题，"
        "并保持语言一致。只输出改写后的问题，不要多余解释。\n\n"
        f"最近对话：\n{ctx}\n\n当前问题：{user_question}"
    )


def format_context(candidates: Sequence[dict]) -> str:
    """把检索到的候选拼成 LLM 上下文，含来源标注（固定 chunk id 以便引用校验）。"""
    block = []
    for i, c in enumerate(candidates, 1):
        doc = c.get("metadata", {}).get("doc_name", "未知文档")
        page = c.get("metadata", {}).get("page_num", 0)
        block.append(f"[{i}] {c.get('content', '')} (来源: {doc}, 第{page}页, chunk_id: {c.get('chunk_id')})")
    return "\n\n".join(block)
