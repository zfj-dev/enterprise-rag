"""运行时组件装配（按 settings 选择 Fake/真实，依赖注入入口）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.config import get_settings
from app.core.bm25 import InMemoryBm25
from app.core.byok import (InMemoryUserLLMConfigStore, LLMFactory,
                           OpenAICompatLLMFactory, UserLLMConfigStore)
from app.core.cache import SemanticCache
from app.core.context import (ApproxTokenCounter, LlmSummarizer, Summarizer,
                               TokenCounter)
from app.core.chunker import ParentChildChunker
from app.core.embedding import EmbeddingModel, get_embedding
from app.core.llm import LLM, get_llm
from app.core.parser import ParserRouter
from app.core.reranker import Reranker, get_reranker
from app.core.retriever import HybridRetriever
from app.core.vector_store import VectorStore, get_vector_store


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
    user_llm_config_store: UserLLMConfigStore = field(default_factory=InMemoryUserLLMConfigStore)
    token_counter: TokenCounter = field(default_factory=ApproxTokenCounter)
    context_summarizer_factory: Callable[[LLM], Summarizer] = LlmSummarizer

    def llm_for(self, user_id: str) -> LLM:
        """按发起用户解析 LLM：配了自带模型就用它，否则回落服务端全局（行为与今天一致）。"""
        cfg = self.user_llm_config_store.get(user_id)
        return self.llm_factory.build(cfg) if cfg else self.llm


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
    return Runtime(embedding=embedding, vector_store=vector_store, bm25=bm25,
                   reranker=reranker, llm=llm, chunker=chunker, parser=parser, retriever=retriever,
                   semantic_cache=semantic_cache)
