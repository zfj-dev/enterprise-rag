"""应用配置（pydantic-settings，读环境变量 / .env）。

USE_REAL=False -> 演示/测试：in-memory 向量库 + Fake 嵌入/重排/LLM（无需 GPU/数据库/Key）
USE_REAL=True  -> 真实模式：bge 嵌入/重排(本机 GPU) + 云端 API LLM（需 Key），在宿主跑
"""
from __future__ import annotations

import logging

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# 托管嵌入/重排的默认站点（SiliconFlow）。两者共用一份，改一家只需改这里；
# 要接别家 OpenAI 兼容的托管服务，用 EMBEDDING_API_BASE / RERANK_API_BASE 覆盖即可。
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

# 仓库里出现过、或一眼可猜的「公开密钥」。拿它们签 JWT = 把管理员身份发给所有人 ——
# 这些值在 README / .env.example / 部署编排 / 运行脚本里都能读到，所以点名拒绝。
# （下面另有一道长度闸兜底：黑名单挡不住随手写的 `my-secret-123`。）
#
# 范围是**部署默认值**，不是「仓库里出现过的每一串」：`tests/conftest.py` 的测试密钥也在
# 仓库里，但它不是任何人的部署值 —— 把它拉黑只会让整套测试起不来。真正的兜底是长度闸。
_PUBLIC_SECRETS = frozenset({
    "dev-secret-change-me-0123456789abcdef",              # 本文件的代码默认值
    "change-me",                                          # .env.example
    "change-me-in-prod",                                  # deploy/docker-compose.yml
    "change-me-strong-0123456789abcdef0123456789",        # deploy/.env.example
    "dev-rag-secret-0123456789abcdef0123456789",          # scripts/run_real.ps1.example
    "changeme", "secret", "test", "password",
})

# 密钥最短长度。黑名单只能挡住**已经公开**的那几个，挡不住随手写的 `my-secret-123`；
# 长度闸是兜底。32 字符 ≈ 192 bit，足够抗离线爆破。
_MIN_SECRET_CHARS = 32

# 演示模式的管理员初始口令。**真实模式不用它**（要么配 ADMIN_PASSWORD，要么随机生成）。
# 放在这里而不是 main.py：服务端与几个客户端脚本（selftest / eval_http）要共用同一个值，
# 「这个口令是公开的」这件事只该写一次。
DEMO_ADMIN_PASSWORD = "admin123"


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
    # 管理员 admin 的**初始**口令（只在该用户还不存在时生效）。
    # ⚠️ 必须走配置字段而不是裸读 `os.environ`：pydantic-settings 会把 `.env` 灌进
    # Settings 对象、**不**写进 `os.environ`；裸读的话，按 .env.example 写进 `.env` 的值
    # 会被静默忽略、回落成公开口令（安全审查的修复第一版就踩了这个坑，与 core/tokenizer.py
    # 里 HF_ENDPOINT 是同一类问题）。
    # 加 `repr=False`：这是口令，随手 `print(settings)` 就泄漏了（同 BYOK 的 key）。
    admin_password: str = Field(default="", repr=False)

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
    # HuggingFace 端点（国内用 https://hf-mirror.com）。**必须走配置**：huggingface_hub 只认
    # 环境变量，而 .env 里的值 pydantic 不进 os.environ —— 加载器会把它补进去（见 core/tokenizer.py）。
    hf_endpoint: str = ""
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
    # 引用覆盖率门槛（票 B）：**逐句校验后**，覆盖率不高于它就改口拒答，不拿模型自己的知识作答。
    # 语义是「必须**严格大于**」—— 默认 0.0 = 一条依据都没有就拒答（这条规则不需要标定）。
    # 调高更严（0.5 = 一半句子要有依据；1.0 = 只要有一句没依据就拒答），
    # 但那是取值，得先在真机上量过再定，别凭空写。
    citation_min_coverage: float = 0.0

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
    # 裁判逐条判定的并发度（#55）：一条一问把正确性换回来了，但调用次数 ×N，
    # 而这些判定彼此独立、串行等网络就是白等。设 1 退回串行（排障用）。
    ragas_judge_concurrency: int = 4
    # 裁判两次调用之间的最小间隔（秒）。百炼按「每分钟请求数 + 秒级突发」限流，而裁判
    # 一条问答就能发出几十次小调用 —— 并发一起打就是一波突发（实测被 429 挡）。
    # 评测是离线的，慢一点无所谓；设 0 = 不限速。
    ragas_judge_min_interval: float = 1.2
    semantic_cache_threshold: float = 0.92
    redis_url: str | None = None
    login_rate_limit_per_min: int = 10  # 登录限流:每用户名每分钟最多尝试次数,超限 429
    # 注册限流（**按 IP**）：注册接口不鉴权，不限流就等于开放建号 —— 每个号都能烧
    # 服务端的 LLM 额度（安全审查 F5）。演示/测试若一次建很多号，把这个调大。
    register_rate_limit_per_hour: int = 10
    # 问答限流（**按用户**）：`max_concurrent_streams_per_user` 管的是「同时几个流」，
    # 管不住「一个接一个地发」—— 后者才是烧钱的方式（安全审查 F5）。
    chat_rate_limit_per_min: int = 30

    max_upload_mb: int = 50
    # **非上传**接口的请求体上限（MB）。字段级的 max_length 不省内存（校验在 body 读完
    # 之后），所以必须有这一道读前拦截；直连 uvicorn 的部署形态只有它能挡（安全审查 F1）。
    max_body_mb: int = 1
    upload_dir: str = "./uploaded_files"
    data_dir: str = "./data"

    @model_validator(mode="after")
    def _enforce_strong_secret(self):
        """**任何模式**都不许拿公开密钥签 JWT。

        原来只在 `USE_REAL=true` 时拦，而 `deploy/docker-compose.yml` 恰好是 `USE_REAL=false`
        —— 于是那份「开箱即用」的编排带着仓库里公开的 SECRET_KEY 起来，任何人
        `jwt.encode({"sub": "admin", "role": "admin"}, "change-me-in-prod")` 就能伪造管理员令牌。
        密钥的强度与「调不调真实模型」无关，所以这道闸不再看 use_real。
        """
        if self.secret_key.strip() in _PUBLIC_SECRETS:
            raise ValueError(
                "SECRET_KEY 用了仓库里公开的默认值 —— 任何人都能伪造 JWT。"
                "请生成一个随机串：python -c \"import secrets;print(secrets.token_urlsafe(48))\""
            )
        if len(self.secret_key.strip()) < _MIN_SECRET_CHARS:
            raise ValueError(
                "SECRET_KEY 太短（至少 %d 字符）—— 短密钥可被离线爆破，伪造出的 JWT 无法分辨。"
                "请生成一个随机串：python -c \"import secrets;print(secrets.token_urlsafe(48))\""
                % _MIN_SECRET_CHARS
            )
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
