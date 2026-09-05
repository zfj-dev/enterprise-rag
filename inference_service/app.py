"""Private inference node: bge embedding + bge reranker on a cloud GPU (or any GPU box).

Runs on the GPU box, listens on a PRIVATE network only (reachable by the orchestrator host).
Replaces the local RTX 4050 dependency. ASCII only (PowerShell GBK caution).

Run:  python -m uvicorn app:app --host 0.0.0.0 --port 9000
Env:  INFERENCE_TOKEN (shared secret; empty = no auth)
      EMBEDDING_MODEL (default BAAI/bge-large-zh-v1.5), RERANKER_MODEL (default BAAI/bge-reranker-large)
      INFER_DEVICE (cuda/cpu), INFER_CONCURRENCY (max parallel GPU inference, default 4)
"""
from __future__ import annotations

import hashlib
import os
import threading

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="rag-inference-node", version="1.0")

TOKEN = os.environ.get("INFERENCE_TOKEN", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-large")
DEVICE = os.environ.get("INFER_DEVICE", "cuda")
CONCURRENCY = int(os.environ.get("INFER_CONCURRENCY", "4"))

_embed_model = None
_rerank_model = None
_rerank_tok = None
_model_lock = threading.Lock()
_sem = threading.Semaphore(CONCURRENCY)
_cache: dict[str, list[float]] = {}
_cache_lock = threading.Lock()


def _enter_authed(x_token: str | None) -> None:
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "invalid inference token")


def _get_embed():
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
            if DEVICE == "cuda":
                try:
                    _embed_model = _embed_model.half()
                except Exception:
                    pass
        return _embed_model


def _get_rerank():
    global _rerank_model, _rerank_tok
    with _model_lock:
        if _rerank_model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            _rerank_tok = AutoTokenizer.from_pretrained(RERANKER_MODEL)
            _rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
            _rerank_model.to(DEVICE)
            _rerank_model.eval()
            if DEVICE == "cuda":
                try:
                    _rerank_model = _rerank_model.half()
                except Exception:
                    pass
            _rerank_model._torch = torch
        return _rerank_model, _rerank_tok


class EmbedReq(BaseModel):
    texts: list[str]


class RerankReq(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "embed_loaded": _embed_model is not None,
            "rerank_loaded": _rerank_model is not None, "device": DEVICE}


@app.post("/embed")
def embed(req: EmbedReq, x_token: str | None = Header(default=None)):
    _enter_authed(x_token)
    model = _get_embed()
    texts = req.texts or []
    vectors: list[list[float]] = []
    with _cache_lock:
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).hexdigest()
            if h in _cache:
                vectors.append(_cache[h])
            else:
                vectors.append(None)  # mark miss, fill below
    # fill misses via batched encode
    miss_idx = [i for i, v in enumerate(vectors) if v is None]
    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        emb = model.encode(miss_texts, normalize_embeddings=True, batch_size=32,
                           show_progress_bar=False).tolist()
        for i, vec in zip(miss_idx, emb):
            vectors[i] = vec
        with _cache_lock:
            for i, vec in zip(miss_idx, emb):
                _cache[hashlib.md5(texts[i].encode("utf-8")).hexdigest()] = vec
        # trim cache to avoid unbounded growth
        if len(_cache) > 200000:
            with _cache_lock:
                if len(_cache) > 200000:
                    _cache.clear()
    return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0}


@app.post("/rerank")
def rerank(req: RerankReq, x_token: str | None = Header(default=None)):
    _enter_authed(x_token)
    docs = req.documents or []
    if not docs:
        return {"results": []}
    model, tok = _get_rerank()
    torch = model._torch
    with _sem:
        pairs = [(req.query, d) for d in docs]
        enc = tok(pairs, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            scores = model(**enc).logits.squeeze(-1).tolist()
    if not isinstance(scores, list):
        scores = [scores]
    top_n = req.top_n or len(docs)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    results = [{"index": i, "relevance_score": float(scores[i])} for i in order[:top_n]]
    return {"results": results}
