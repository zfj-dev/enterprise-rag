"""Pydantic 请求/响应模型（API 层）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Auth ----
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, description="用户名(1-64字符)")
    password: str = Field(min_length=6, max_length=128, description="密码(6-128字符)")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


# ---- Knowledge base ----
class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: str
    embedding_model: str
    doc_count: int = 0


# ---- Documents ----
class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int
    error: str = ""
    progress: int = 100  # 处理进度 0-100（仅处理中有效）
    size: int = 0  # 文件大小(字节)
    created_at: str = ""  # 上传时间 ISO


# ---- Chat ----
class ChatRequest(BaseModel):
    kb_id: str
    question: str
    session_id: str | None = None
    stream: bool = True


class SourceOut(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)


# ---- Feedback ----
class FeedbackRequest(BaseModel):
    message_id: str
    rating: int  # 1 赞 / -1 踩
    comment: str = ""


class FeedbackOut(BaseModel):
    id: str
    rating: int
    comment: str


# ---- Debug ----
class DebugTraces(BaseModel):
    query: str
    rewrite: str | None = None
    retrieval_top: list[dict] = Field(default_factory=list)
    reranked: list[dict] = Field(default_factory=list)
    answer: str = ""


# ---- Metrics ----
class MetricsOut(BaseModel):
    faithfulness: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    citation_coverage: float | None = None
    total_answered: int = 0
