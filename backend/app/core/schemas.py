"""Pydantic 请求/响应模型（API 层）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Auth ----
class LoginRequest(BaseModel):
    # 登录：不做强度/长度校验（短密码=历史老账号，应返回401而非422；避免泄露校验信息）
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class RegisterRequest(BaseModel):
    # 注册：校验强度（用户名1-64、密码6-128）
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)


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


class CostBucketOut(BaseModel):
    """一桶成本：按人（`user_id` / `username`）或按天（`date`）。"""

    key: str = ""                              # 展示名：用户名 / 日期
    user_id: str | None = None
    date: str | None = None
    cost: float | None = None                  # 折得出来的合计；**一笔都折不出就是 None**
    priced: int = 0
    unpriced: int = 0                          # 折不出费用的条数 —— 所以 cost 是下界
    records: int = 0


class CostSummaryOut(BaseModel):
    """成本摘要：**管理员看全局，普通用户只看自己**（`group` 标明这次给的是哪一种）。"""

    group: str = ""                            # global（管理员）/ self（普通用户）
    window: str = ""                           # 自然窗口（day / month）或滚动范围
    since: str = ""                            # 时间窗起点（本地时间）
    days: int | None = None                    # 只在滚动范围时有值
    records: int = 0
    total_cost: float | None = None            # 同上：折不出就是 None，不是 0
    unpriced: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_unavailable: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)   # 账单 / 估算 / 模拟 / 不可用 各几笔
    by_user: list[CostBucketOut] = Field(default_factory=list)   # 只有管理员看得到
    by_day: list[CostBucketOut] = Field(default_factory=list)
    note: str = ""


# ---- 自带 Key（BYOK）----
class LLMConfigIn(BaseModel):
    """用户填进来的三样。**key 只在这一个方向出现** —— 出去的时候只剩尾号。"""

    base_url: str = Field(min_length=1)
    key: str = Field(min_length=1, repr=False)   # 不进 repr：日志里顺手打一下就泄漏了
    model: str = Field(min_length=1)


class LLMConfigOut(BaseModel):
    """回显：**只有非敏感信息**。字段里压根没有 key 的位置。"""

    configured: bool = False
    base_url: str = ""
    model: str = ""
    key_tail: str = ""          # 尾号，排障够用，泄漏不了
    updated_at: str = ""
    persistent: bool = False    # 重启后还在不在 —— 只存内存时如实说「不在」，别让人以为存住了


# ---- 跨会话记忆 ----
class MemoryFactOut(BaseModel):
    id: str
    content: str
    session_id: str = ""
