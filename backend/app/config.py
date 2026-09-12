"""应用配置（pydantic-settings，读环境变量 / .env）。

USE_REAL=False -> 演示/测试：in-memory 向量库 + Fake 嵌入/重排/LLM（无需 GPU/数据库/Key）
USE_REAL=True  -> 真实模式：bge 嵌入/重排(本机 GPU) + 云端 API LLM（需 Key），在宿主跑
"""
from __future__ import annotations

import logging

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# 托管嵌入/重排的默认站点（SiliconFlow）。两者共用一份，改一家只需改这里；
# 要接别家 OpenAI 兼容的托管服务，用 EMBEDDING_API_BASE / RERANK_API_BASE 覆盖即可。
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"


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

    cors_origins: str = "*"  # 允许跨域来源,逗号分隔; 默认* = 局域网 demo,生产用 CORS_ORIGINS=http://a,http://b
    max_concurrent_streams_per_user: int = 2  # 每用户同时流式对话上限,防单客户端打爆后台 LLM/GPU

    vector_store: Literal["inmemory", "pgvector"] = "inmemory"

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    embedding_provider: Literal["fake", "bge", "api", "siliconflow"] = "fake"
    embedding_device: str = "cuda"  # bge 用；无 GPU 会自动回落 cpu
    # 嵌入走外部服务时的站点：自建推理节点(api) 与托管(siliconflow) 都用它；
    # 留空则用托管默认站点
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    embedding_batch_size: int = 64

    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_enabled: bool = True
    # 严格模式（**评测用**）：重排不可达时**直接抛**，不降级为 RRF 原顺序。
    # 线上保持默认 false —— 重排挂掉不该把问答也打断；但评测要的是「这段没跑成」，
    # 不是把 RRF 顺序的数字说成「重排已跑」（票 39 / #48）。
    rerank_strict: bool = False
    reranker_provider: Literal["fake", "bge", "api", "siliconflow"] = "fake"
    reranker_device: str = "cuda"
    # 重排走外部服务时的站点，同上
    rerank_api_base: str | None = None
    rerank_api_key: str | None = None

    llm_provider: Literal["fake", "deepseek", "siliconflow", "openai", "dashscope"] = "fake"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048  # 枚举/长回答不截断
    # 上下文预算（读取侧）：历史超过此 token 数才压缩。**与 llm_max_tokens（输出上限）无关**。
    context_token_budget: int = 3000
    context_compress: bool = True    # 关闭则不做压缩，退回只带最近若干轮原文
    context_keep_recent: int = 3        # 压缩时保留的最近轮数（原文）
    # 真实分词器（票 19）：填**与生成模型同族**的模型名（Qwen 链路如 Qwen2.5-7B-Instruct）。
    # 默认**留空** = 不接：演示模式没有"同族"可言，也免得每次启动都去打网下载；
    # 真实模式由 run_real.ps1 设好。留空/拿不到时分词器指标一律报「不可用」，绝不回退成字数估算。
    tokenizer_model: str = ""
    context_history_messages: int = 20  # 每次问答从库里加载的历史**消息**条数上限（一轮=2 条）
    # 票 18 已知边界：加载窗口外的轮次压根不进装配，也就不会被滚进摘要 —— 对话涨得比
    # 预算快时，窗口外更早的轮次恢复不了（要更早的历史就只能调大这个窗口）
    fake_llm_delay: float = 0.0  # FakeLLM 每块延时(秒)，默认0不延时；设>0 便于演示/测试肉眼观察流式与"停止"

    chunk_parent_size: int = 512
    chunk_child_size: int = 128
    chunk_overlap: int = 32
    contextual_summary: bool = True

    retrieval_top_k: int = 20
    rerank_top_k: int = 3  # 上下文/来源宽度（具体查询更聚焦）
    min_relevance: float = 0.4
    rrf_k: int = 60

    parser_use_docling: bool = True  # 装了 docling 且 PDF 走它(表格/版面更好)，否则回退 PyMuPDF
    docling_images_scale: float = 1.0  # docling 版面分析图像倍率；0.5=更快但小表格/图可能漏
    docling_table_mode: str = "accurate"  # accurate=更准/fast=更快
    docling_formula_enrichment: bool = False  # 锁定=关：Docling 能从文本层抽原始公式文本($$..$$)，企业文档公式少，不上 CodeFormulaV2 VLM
    docling_formula_ocr: bool = True  # 公式图片→LaTeX：用 pix2tex 裁图识别(轻量,CPU可跑)；未装 pix2tex 或识别失败自动跳过
    semantic_cache: bool = True
    # 代理链路开关（票 15）。**默认关**：关着时问答走原确定性管线，行为与今天完全一致。
    agent_enabled: bool = False
    agent_max_steps: int = 4      # ReAct 循环步数上限（失控时能停下、不烧钱）
    memory_enabled: bool = True   # 跨会话记忆：问答后异步抽取"用户告知的事实"并按用户落库
    cost_enabled: bool = True     # 用量记账（票 27）：每次生成记一条 token 用量与口径来源
    # 价格表覆盖（票 28）：JSON，形如 {"qwen-plus": {"input": 0.0008, "output": 0.002}}（元/1K token）。
    # 内置价只是**参考价**、会过期；表里没有的模型一律标「单价未知」而**不按 0 算**。
    llm_price_overrides: str = ""
    # BYOK（票 32）：用户自带 Key。加密口令**没有默认值、也没有弱默认** —— 没配就不落库
    # （凭据只存内存、重启即失），绝不退化成明文存储。
    byok_secret_key: str = ""
    # 自填 base_url 的三条线（票 33）：白名单（逗号分隔，空=不限主机）、显式放行明文 http
    # （只在本地开发用）、以及**请求超时上限**（用户填的地址不由我们控制，超时必须封顶）。
    byok_allowed_hosts: str = ""
    byok_allow_insecure: bool = False
    byok_request_timeout_seconds: float = 30.0
    # 厂商余额的**提醒线**（票 35）：低于它只提醒、不拦截；<=0 表示不设提醒
    byok_balance_alert_threshold: float = 0.0
    # 预算硬拦（票 29）。**默认关**：硬拦会挡住用户，先让人显式打开。
    quota_enabled: bool = False
    quota_window: Literal["day", "month"] = "day"   # 自然窗口（日 / 月）
    quota_limit: float = 0.0                        # 每个用户每窗口的费用上限（元）；<=0 = 未设上限
    memory_extract_max_facts: int = 5  # 单轮最多抽取几条事实
    memory_recall_top_k: int = 3          # 每次问答最多注入几条记忆（有上限，不堆爆上下文）
    memory_recall_min_score: float = 0.35  # 相似度低于此不注入：无相关记忆时零注入
    memory_inject_max_chars: int = 200     # 单条事实**注入时**的长度上限（存储不截断）
    ragas_judge_model: str = "qwen-turbo"  # RAGAS 裁判固定口径：模型写进报告，数字才可跨时间比较
    ragas_judge_temperature: float = 0.0
    semantic_cache_threshold: float = 0.92
    redis_url: str | None = None
    login_rate_limit_per_min: int = 10  # 登录限流:每用户名每分钟最多尝试次数,超限 429

    max_upload_mb: int = 50
    upload_dir: str = "./uploaded_files"
    data_dir: str = "./data"

    @model_validator(mode="after")
    def _enforce_secret_in_real(self):
        """真实模式必须用强 SECRET_KEY，避免用默认 dev 值伪造 JWT。"""
        if self.use_real and self.secret_key == "dev-secret-change-me-0123456789abcdef":
            raise ValueError("真实模式(USE_REAL=true)必须设置强 SECRET_KEY 环境变量，不能使用默认值")
        return self

    @model_validator(mode="after")
    def _warn_byok_without_an_encryption_key(self):
        """没配加密口令时凭据只存内存（重启即失）—— 说出来，别让人以为已经落库了。"""
        if self.byok_secret_key:
            return self
        logger.warning("未配置 BYOK_SECRET_KEY：自带 Key **不落库**（只存内存，重启即失）。"
                       "要跨会话保存请设一个足够强的口令。")
        return self

    @model_validator(mode="after")
    def _warn_quota_without_a_limit(self):
        """开了额度硬拦却没设上限 = 谁都不会被拦 —— 「开了等于没开」要说出来，别让人以为已经在管。"""
        if self.quota_enabled and self.quota_limit <= 0:
            logger.warning("QUOTA_ENABLED=true 但 QUOTA_LIMIT<=0：硬拦开着却拦不住任何人，请设上限")
        return self

    def all_llm_url(self) -> str:
        if self.llm_provider == "deepseek":
            return self.llm_base_url or "https://api.deepseek.com/v1"
        if self.llm_provider == "siliconflow":
            return self.llm_base_url or SILICONFLOW_BASE
        if self.llm_provider == "dashscope":
            return self.llm_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return self.llm_base_url or "https://api.openai.com/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
