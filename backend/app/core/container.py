"""运行时组件装配（按 settings 选择 Fake/真实，依赖注入入口）。"""
from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Callable

from app.config import get_settings
from app.core.bm25 import InMemoryBm25
from app.core.byok import (DbUserLLMConfigStore, InMemoryUserLLMConfigStore, LLMFactory,
                           OpenAICompatLLMFactory, UserLLMConfigStore)
from app.core.cache import SemanticCache
from app.core.context import (ApproxTokenCounter, LlmSummarizer, Summarizer,
                               TokenCounter)
from app.core.chunker import ParentChildChunker
from app.core.embedding import EmbeddingModel, get_embedding
from app.core.llm import LLM, get_llm
from app.core.memory import (DbMemoryStore, FactExtractor, LlmFactExtractor,
                             MemoryRecall, MemoryStore)
from app.core.parser import ParserRouter
from app.core.pricing import PriceTable
from app.core.reranker import Reranker, get_reranker
from app.core.retriever import HybridRetriever
from app.core.usage import DbUsageStore, UsageStore
from app.core.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    embedding: EmbeddingModel
    vector_store: VectorStore
    bm25: InMemoryBm25
    reranker: Reranker
    llm: LLM
    chunker: ParentChildChunker
    parser: ParserRouter
    retriever: HybridRetriever
    semantic_cache: SemanticCache
    llm_factory: LLMFactory = field(default_factory=OpenAICompatLLMFactory)
    # 默认内存实现（谁都没配过 → 全部回落全局）；配了 BYOK_SECRET_KEY 才换加密落库那一个
    user_llm_config_store: UserLLMConfigStore = field(default_factory=InMemoryUserLLMConfigStore)
    token_counter: TokenCounter = field(default_factory=ApproxTokenCounter)
    context_summarizer_factory: Callable[[LLM], Summarizer] = LlmSummarizer
    fact_extractor_factory: Callable[[LLM], FactExtractor] = LlmFactExtractor
    memory_store: MemoryStore = field(default_factory=DbMemoryStore)
    usage_store: UsageStore = field(default_factory=DbUsageStore)
    price_table: PriceTable = field(default_factory=PriceTable)

    def llm_for(self, user_id: str) -> LLM:
        """按发起用户解析 LLM：配了自带模型就用它，否则回落服务端全局（行为与今天一致）。"""
        cfg = self.user_llm_config_store.get(user_id)
        return self.llm_factory.build(cfg) if cfg else self.llm

    def recall_memory(self, user_id: str, question: str) -> list[dict]:
        """按相似度召回该用户的相关记忆（供注入；不进检索候选池、不作引用来源）。

        刻意做成方法、而不是在 build_runtime 里装配成字段：记忆存储与嵌入都是
        测试要替换的缝，方法每次现取，替换 rt.memory_store / rt.embedding 立即生效；
        装配成字段则会抓住装配时的旧引用。
        """
        return MemoryRecall(store=self.memory_store, embedding=self.embedding).recall(user_id, question)


def build_runtime() -> Runtime:
    s = get_settings()
    embedding = get_embedding()
    vector_store = get_vector_store(backend=s.vector_store, conn_url=s.database_url, dim=s.embedding_dim)
    bm25 = InMemoryBm25()
    reranker = get_reranker()
    llm = get_llm()
    chunker = ParentChildChunker()
    parser = ParserRouter()
    retriever = HybridRetriever(vector_store=vector_store, bm25=bm25, embedding=embedding, reranker=reranker)
    semantic_cache = SemanticCache(embedding=embedding, backend="redis" if s.redis_url else "memory")
    from app.core.tokenizer import setup_token_counter

    token_counter, _ = setup_token_counter(s.tokenizer_model)   # 拿不到就回落估算（只是预算用）
    price_table = PriceTable(s.llm_price_overrides)
    for issue in price_table.warnings:
        # 启动时就说出来：静默回落内置价，会让人拿到一个看着正常的费用却不知道配置没生效
        logger.warning("价格表配置有问题：%s", issue)
    # 凭据存储：**配了口令才加密落库**；没配就只存内存（重启即失），绝不退化成明文
    byok_store = (DbUserLLMConfigStore(s.byok_secret_key) if s.byok_secret_key
                  else InMemoryUserLLMConfigStore())
    return Runtime(embedding=embedding, vector_store=vector_store, bm25=bm25,
                   user_llm_config_store=byok_store,
                   reranker=reranker, llm=llm, chunker=chunker, parser=parser, retriever=retriever,
                   semantic_cache=semantic_cache, token_counter=token_counter,
                   price_table=price_table)
