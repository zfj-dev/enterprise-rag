"""运行时组件装配（按 settings 选择 Fake/真实，依赖注入入口）。"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.core.bm25 import InMemoryBm25
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
    return Runtime(embedding=embedding, vector_store=vector_store, bm25=bm25,
                   reranker=reranker, llm=llm, chunker=chunker, parser=parser, retriever=retriever)
