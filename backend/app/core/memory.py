"""跨会话记忆：抽取 + 落库（票 23）、召回 + 注入块正文（票 24）。

抽取器、存储与嵌入都从外部注入 —— 测试因而无网络、无真实 LLM。
面向用户的 HTTP 接口在 `app/api/v1/memory.py`。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import get_settings
from app.core.llm import LLM
from app.core.similarity import cosine
from app.utils.text import lines_of
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
        for t in lines_of(raw):        # 拆行 + 去列表标记（与评测判据共用同一份口径）
            if t not in facts:
                facts.append(t)
        return facts[: self._max]


class MemoryStore(ABC):
    """按用户存取与删除记忆事实。全部按 user 下推过滤，取不到他人的。"""

    @abstractmethod
    def add(self, user_id: str, facts: Sequence[str], *, session_id: str = "",
            message_id: str = "") -> int: ...

    @abstractmethod
    def list(self, user_id: str) -> list[dict]: ...

    @abstractmethod
    def delete(self, user_id: str, fact_id: str) -> bool:
        """删掉该用户的某条记忆；不是他的（或不存在）返回 False。"""


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

    def delete(self, user_id, fact_id):
        bucket = self._by_user.get(user_id, [])
        for i, f in enumerate(bucket):
            if f["id"] == fact_id:
                del bucket[i]
                return True
        return False


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
            # created_at 由 ORM 侧单调默认值保证严格递增（entities._monotonic_utc_now），
            # id 只是本次改动之前写入的同秒旧行的确定性兜底。
            rows = (db.query(MemoryFact).filter(MemoryFact.user_id == user_id)
                    .order_by(MemoryFact.created_at.asc(), MemoryFact.id.asc()).all())
            return [{"id": r.id, "content": r.content, "session_id": r.source_session_id,
                     "message_id": r.source_message_id} for r in rows]
        finally:
            db.close()

    def delete(self, user_id, fact_id):
        db = SessionLocal()
        try:
            row = (db.query(MemoryFact)
                   .filter(MemoryFact.id == fact_id, MemoryFact.user_id == user_id).first())
            if not row:
                return False
            db.delete(row)
            db.commit()
            return True
        finally:
            db.close()


def _truncate(text: str, limit: int) -> str:
    """超长事实截断时补省略号 —— 注入半句事实会读成另一句，比注入稍短的事实更糟。"""
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


class MemoryRecall:
    """按相似度从某用户的记忆里召回 top-k 条相关事实。

    复用现有嵌入缝（不新增相似度算法）。store 与 embedding 在调用时现取 ——
    测试替换 Runtime 上的这两个缝，即可确定性地测召回。
    """

    def __init__(self, store: MemoryStore, embedding, top_k: int | None = None,
                 min_score: float | None = None, max_chars: int | None = None):
        s = get_settings()
        self._store = store
        self._embedding = embedding
        self._top_k = top_k if top_k is not None else s.memory_recall_top_k
        self._min_score = min_score if min_score is not None else s.memory_recall_min_score
        self._max_chars = max_chars if max_chars is not None else s.memory_inject_max_chars

    def recall(self, user_id: str, question: str) -> list[dict]:
        facts = self._store.list(user_id)
        if not facts or not question:
            return []
        qv = self._embedding.encode([question])[0]
        vecs = self._embedding.encode([f["content"] for f in facts])
        hits = [(cosine(qv, v), f) for f, v in zip(facts, vecs) if v]
        hits = [(sc, f) for sc, f in hits if sc >= self._min_score]
        hits.sort(key=lambda x: x[0], reverse=True)
        return [{"content": _truncate(f["content"], self._max_chars), "score": round(sc, 3)}
                for sc, f in hits[: self._top_k]]
