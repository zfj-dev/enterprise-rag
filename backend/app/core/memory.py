"""跨会话记忆 —— 票 23 只做「抽取 + 按用户落库」。

抽取器与存储都从外部注入 —— 测试因而无网络、无真实 LLM。
**不在**本模块范围：召回与注入（票 24）、面向用户的列出与删除（票 25）。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings
from app.core.llm import LLM
from app.db.session import SessionLocal
from app.models.entities import MemoryFact

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "从下面这轮「用户提问 + 助手回答」中，抽出**用户明确告知的、可跨会话复用的事实**"
    "（例如身份、偏好、所在项目与业务背景、约定的口径）。\n"
    "严格要求：\n"
    "1. 只抽**用户说过的**；**不要**摘抄文档或参考资料的内容，**也不要**把助手自己的回答当成事实。\n"
    "2. 用户没明确说过的不要抽（不要从提问方式里推测）。\n"
    "3. 每条事实占一行，不要编号、不要解释；没有可抽的就输出空行。\n\n"
)


def _strip_list_marker(line: str) -> str:
    """去掉行首的列表标记（`- ` / `* ` / `2. ` / `3) `），**只**去标记，不吞正文。

    不能用 lstrip 把数字和点号统统削掉 —— 那会把「2024 年营收」这类以数字开头的事实削成「年营收」。
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


class FactExtractor(ABC):
    """从一轮问答里抽出「用户告知的事实」。"""

    @abstractmethod
    def extract(self, question: str, answer: str) -> list[str]: ...


class LlmFactExtractor(FactExtractor):
    """用给定 LLM 抽取。构造参数即工厂：`LlmFactExtractor(llm)`。"""

    def __init__(self, llm: LLM, max_facts: int | None = None):
        self._llm = llm
        self._max = max_facts or get_settings().memory_extract_max_facts

    def extract(self, question: str, answer: str) -> list[str]:
        prompt = _EXTRACT_PROMPT + f"【用户提问】\n{question}\n\n【助手回答】\n{answer}"
        raw = "".join(self._llm.stream([{"role": "user", "content": prompt}]))
        facts: list[str] = []
        for line in raw.splitlines():
            t = _strip_list_marker(line)
            if t and t not in facts:
                facts.append(t)
        return facts[: self._max]


class MemoryStore(ABC):
    """按用户存取记忆事实。（删除由票 25 补上。）"""

    @abstractmethod
    def add(self, user_id: str, facts: Sequence[str], *, session_id: str = "",
            message_id: str = "") -> int: ...

    @abstractmethod
    def list(self, user_id: str) -> list[dict]: ...


class InMemoryMemoryStore(MemoryStore):
    """内存实现：测试隔离用。"""

    def __init__(self) -> None:
        self._by_user: dict[str, list[dict]] = {}
        self._seq = 0

    def add(self, user_id, facts, *, session_id="", message_id=""):
        bucket = self._by_user.setdefault(user_id, [])
        for f in facts:
            self._seq += 1
            bucket.append({"id": f"m{self._seq}", "content": f,
                           "session_id": session_id, "message_id": message_id})
        return len(facts)

    def list(self, user_id):
        return list(self._by_user.get(user_id, []))


class DbMemoryStore(MemoryStore):
    """落库实现。自己开 session —— 抽取在后台线程里跑，请求期的 db 那时已关闭。"""

    def add(self, user_id, facts, *, session_id="", message_id=""):
        if not facts:
            return 0
        db = SessionLocal()
        try:
            for f in facts:
                db.add(MemoryFact(user_id=user_id, content=f,
                                  source_session_id=session_id or "",
                                  source_message_id=message_id or ""))
            db.commit()
            return len(facts)
        finally:
            db.close()

    def list(self, user_id):
        db = SessionLocal()
        try:
            rows = (db.query(MemoryFact).filter(MemoryFact.user_id == user_id)
                    .order_by(MemoryFact.created_at.asc(), MemoryFact.id.asc()).all())
            return [{"id": r.id, "content": r.content, "session_id": r.source_session_id} for r in rows]
        finally:
            db.close()
