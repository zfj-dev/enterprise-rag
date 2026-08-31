"""应用配置（pydantic-settings，读环境变量 / .env）。

USE_REAL=False -> 演示/测试：in-memory 向量库 + Fake 嵌入/重排/LLM（无需 GPU/数据库/Key）
USE_REAL=True  -> 真实模式：bge 嵌入/重排(本机 GPU) + 云端 API LLM（需 Key），在宿主跑
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "企业智能文档问答系统"
    api_prefix: str = "/api/v1"

    use_real: bool = False

    database_url: str = "sqlite:///./rag.db"
    db_echo: bool = False

    secret_key: str = "dev-secret-change-me-0123456789abcdef"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    vector_store: Literal["inmemory", "pgvector"] = "inmemory"

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    embedding_provider: Literal["fake", "bge"] = "fake"
    embedding_device: str = "cuda"  # bge 用；无 GPU 会自动回落 cpu

    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_enabled: bool = True
    reranker_provider: Literal["fake", "bge"] = "fake"
    reranker_device: str = "cuda"

    llm_provider: Literal["fake", "deepseek", "siliconflow", "openai", "dashscope"] = "fake"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048  # 枚举/长回答不截断

    chunk_parent_size: int = 512
    chunk_child_size: int = 128
    chunk_overlap: int = 32
    contextual_summary: bool = True

    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    min_relevance: float = 0.4
    rrf_k: int = 60

    parser_use_docling: bool = True  # 装了 docling 且 PDF 走它(表格/版面更好)，否则回退 PyMuPDF
    docling_images_scale: float = 1.0  # docling 版面分析图像倍率；0.5=更快但小表格/图可能漏
    docling_table_mode: str = "accurate"  # accurate=更准/fast=更快
    docling_formula_enrichment: bool = False  # 公式增强(解码公式为LaTeX)，需下 CodeFormulaV2 ~630MB，解析变慢
    semantic_cache: bool = True
    semantic_cache_threshold: float = 0.92
    redis_url: str | None = None

    max_upload_mb: int = 50
    upload_dir: str = "./uploaded_files"
    data_dir: str = "./data"

    def all_llm_url(self) -> str:
        if self.llm_provider == "deepseek":
            return self.llm_base_url or "https://api.deepseek.com/v1"
        if self.llm_provider == "siliconflow":
            return self.llm_base_url or "https://api.siliconflow.cn/v1"
        if self.llm_provider == "dashscope":
            return self.llm_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return self.llm_base_url or "https://api.openai.com/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
