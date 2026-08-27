"""应用配置（pydantic-settings，读环境变量 / .env）。

USE_REAL=False  -> 演示/测试模式：in-memory 向量库 + Fake 嵌入/重排/LLM（无需 GPU/数据库/Key）
USE_REAL=True   -> 真实模式：pgvector + bge 本地(需 GPU) + 云端 API LLM（需 Key），见 .env.example
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "企业智能文档问答系统"
    api_prefix: str = "/api/v1"

    # 运行模式开关
    use_real: bool = False  # False=fake/demo（开发与测试）；True=真实组件

    # 数据库
    database_url: str = "sqlite:///./rag.db"  # 真实: postgresql+psycopg://rag:rag@postgres:5432/rag_db
    db_echo: bool = False

    # 鉴权
    secret_key: str = "dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # 向量库
    vector_store: Literal["inmemory", "pgvector"] = "inmemory"

    # Embedding（本地 GPU）
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    embedding_provider: Literal["fake", "bge"] = "fake"

    # Reranker（本地 GPU）
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_enabled: bool = True
    reranker_provider: Literal["fake", "bge"] = "fake"

    # LLM（云端 API）
    llm_provider: Literal["fake", "deepseek", "siliconflow", "openai"] = "fake"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # 分块
    chunk_parent_size: int = 512  # 父块（生成用）
    chunk_child_size: int = 128   # 子块（检索用）
    chunk_overlap: int = 32
    contextual_summary: bool = True  # 每个 chunk 前置来源上下文摘要

    # 检索
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    min_relevance: float = 0.4   # 需用真实数据校准
    rrf_k: int = 60

    # 上传限制
    max_upload_mb: int = 50
    upload_dir: str = "./uploaded_files"
    data_dir: str = "./data"

    def all_llm_url(self) -> str:
        if self.llm_provider == "deepseek":
            return self.llm_base_url or "https://api.deepseek.com/v1"
        if self.llm_provider == "siliconflow":
            return self.llm_base_url or "https://api.siliconflow.cn/v1"
        return self.llm_base_url or "https://api.openai.com/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
