"""ORM 实体：用户 / 知识库 / 文档 / 分块 / 会话 / 消息 / 反馈。

id 用 str(uuid4) 便于 sqlite 与 postgres 通用；向量索引单独放 VectorStore，不在此表。
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex


_clock_lock = threading.Lock()
_last_us = 0


def _monotonic_utc_now() -> datetime:
    """严格递增的 UTC 时间戳（微秒精度）—— created_at 的 ORM 侧默认值。

    为什么要单调：sqlite 的 `CURRENT_TIMESTAMP` 只有**秒**级精度，同秒写入的多行
    created_at 全相等 —— 排序退化成随机主键；更糟的是 sqlite 对相等的排序键 ASC/DESC
    都按 rowid 返回（DESC 并不翻转），于是「倒序取最近 N 条再 reversed()」的取法会
    整体错开一格（见 chat_service._load_history）。而 Windows 上 `datetime.now()` 的
    粒度可达 ~15ms，一批插入常拿到完全相同的值，所以这里对时钟做单调修正：
    同刻再取一次就 +1µs。

    只影响经 ORM 的插入，不改既有表结构；`server_default=func.now()` 仍是原始 SQL
    插入的兜底。旧行的朴素时间戳（`...:SS`）是新行（`...:SS.ffffff`）的前缀，
    字符串比较下旧行仍排在前面，混排不会乱序。
    """
    global _last_us
    with _clock_lock:
        us = time.time_ns() // 1000
        if us <= _last_us:
            us = _last_us + 1
        _last_us = us
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=us)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default="viewer")  # admin/uploader/viewer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    embedding_model: Mapped[str] = mapped_column(String(64), default="BAAI/bge-large-zh-v1.5")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())

    documents: Mapped[list["Document"]] = relationship(back_populates="kb")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kb_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    owner_id: Mapped[str] = mapped_column(String(32), index=True)
    filename: Mapped[str] = mapped_column(String(256))
    file_path: Mapped[str] = mapped_column(String(512), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/processing/indexed/failed
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())

    kb: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="doc", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    parent_id: Mapped[str | None] = mapped_column(String(32), index=True, default=None)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    kb_id: Mapped[str] = mapped_column(String(32), index=True)
    owner_id: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)          # child: 检索用
    parent_content: Mapped[str | None] = mapped_column(Text, default=None)  # parent: 生成用
    page_num: Mapped[int] = mapped_column(Integer, default=0)
    chunk_type: Mapped[str] = mapped_column(String(16), default="child")  # parent/child
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())

    doc: Mapped["Document"] = relationship(back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    kb_id: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    # 滚动摘要（票 18）：更早的对话压成一段，随会话推进增量更新。
    # summary_upto = 已摘要到的最后一条消息 id（游标）—— 刷新/重启后靠它接着滚，不从头再来。
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_upto: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user/assistant
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON: [{doc_id,doc_name,page}]
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())


class MemoryFact(Base):
    """跨会话记忆：一条"用户告知过的事实"。按 user 隔离；不作为引用来源。"""

    __tablename__ = "memory_facts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    source_session_id: Mapped[str] = mapped_column(String(32), default="")
    source_message_id: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("chat_messages.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    rating: Mapped[int] = mapped_column(Integer)  # 1=赞 -1=踩
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_monotonic_utc_now, server_default=func.now())
