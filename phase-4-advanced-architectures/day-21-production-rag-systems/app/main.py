"""
Production RAG System — FastAPI Application
Day 21 of the 21-Day RAG Series

Endpoints:
  POST /query          — main RAG query endpoint
  POST /query/batch    — batch query endpoint
  GET  /health         — system health check
  GET  /cache/stats    — cache statistics
  DELETE /cache        — clear cache
  POST /evaluate       — run evaluation pipeline
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.retriever import Retriever
from app.reranker import rerank
from app.context_builder import build_context
from app.prompt import build_rag_prompt
from app.llm import generate, health_check
from app import cache
from app.evaluator import run_evaluation
from app.logger import get_logger

logger = get_logger("main")

# ---------------------------------------------------------------------------
# Global retriever (loaded once at startup)
# ---------------------------------------------------------------------------
retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Starting RAG system...")
    retriever = Retriever()
    logger.info("RAG system ready.")
    yield
    logger.info("Shutting down RAG system.")


app = FastAPI(
    title="Production RAG System",
    description="Day 21 — 21-Day RAG Series | Local Mistral via Ollama",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="User question")
    top_k: int = Field(default=10, ge=1, le=50, description="Retrieval pool size")
    top_n: int = Field(default=4, ge=1, le=20, description="Chunks passed to LLM after reranking")
    keyword_filter: Optional[str] = Field(default="", description="Optional keyword pre-filter")
    use_multi_query: bool = Field(default=False, description="Generate query variations for better recall")
    use_cache: bool = Field(default=True, description="Use cached response if available")


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks_used: int
    sources: list[str]
    latency_s: float
    cache_hit: bool


class BatchQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_items=1, max_items=20)
    top_k: int = Field(default=10)
    top_n: int = Field(default=4)


# ---------------------------------------------------------------------------
# Core pipeline function (reusable by API + evaluator)
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    top_k: int = 10,
    top_n: int = 4,
    keyword_filter: str = "",
    use_multi_query: bool = False,
    use_cache: bool = True
) -> dict:
    start = time.time()

    # 1. Retrieve
    if use_multi_query:
        variations = [
            query,
            query + " explanation",
            "what is " + query,
            "how does " + query + " work",
        ]
        chunks = retriever.multi_query_retrieve(variations, top_k=top_k)
    else:
        chunks = retriever.retrieve(query, top_k=top_k, keyword_filter=keyword_filter)

    if not chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "chunks": [],
            "latency": round(time.time() - start, 3)
        }

    # 2. Rerank
    reranked = rerank(query, chunks, top_n=top_n)

    # 3. Build context
    context = build_context(reranked)

    # 4. Cache check
    cache_key = cache.make_key(query, context)
    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            return {
                "answer": cached,
                "chunks": reranked,
                "latency": round(time.time() - start, 3),
                "cache_hit": True
            }

    # 5. Build prompt
    prompt = build_rag_prompt(context, query)

    # 6. LLM inference
    answer = generate(prompt)

    # 7. Cache response
    if use_cache:
        cache.set(cache_key, answer)

    latency = round(time.time() - start, 3)
    logger.info(
        f"Pipeline complete | query='{query[:60]}' | "
        f"chunks={len(reranked)} | latency={latency}s"
    )

    return {
        "answer": answer,
        "chunks": reranked,
        "latency": latency,
        "cache_hit": False
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Main RAG query endpoint."""
    if retriever is None or retriever.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run scripts/ingest.py first.")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_pipeline(
            query=req.query,
            top_k=req.top_k,
            top_n=req.top_n,
            keyword_filter=req.keyword_filter or "",
            use_multi_query=req.use_multi_query,
            use_cache=req.use_cache
        )
    )

    sources = list({c.get("source", "unknown") for c in result["chunks"]})

    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        chunks_used=len(result["chunks"]),
        sources=sources,
        latency_s=result["latency"],
        cache_hit=result.get("cache_hit", False)
    )


@app.post("/query/batch")
async def batch_query_endpoint(req: BatchQueryRequest):
    """Batch query endpoint — processes multiple queries concurrently."""
    if retriever is None or retriever.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, lambda q=q: run_pipeline(q, req.top_k, req.top_n))
        for q in req.queries
    ]
    results = await asyncio.gather(*tasks)

    return [
        {
            "query": q,
            "answer": r["answer"],
            "chunks_used": len(r["chunks"]),
            "latency_s": r["latency"]
        }
        for q, r in zip(req.queries, results)
    ]


@app.get("/health")
async def health():
    """System health check."""
    llm_ok = health_check()
    index_ok = retriever is not None and retriever.index is not None
    index_size = retriever.index.ntotal if index_ok else 0

    return {
        "status": "ok" if (llm_ok and index_ok) else "degraded",
        "llm_available": llm_ok,
        "index_loaded": index_ok,
        "index_vectors": index_size,
        "cache_stats": cache.stats()
    }


@app.get("/cache/stats")
async def cache_stats():
    return cache.stats()


@app.delete("/cache")
async def clear_cache():
    count = cache.clear()
    return {"message": f"Cache cleared", "entries_removed": count}


@app.post("/evaluate")
async def evaluate(background_tasks: BackgroundTasks):
    """Run evaluation pipeline in the background."""
    if retriever is None or retriever.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    def _pipeline_fn(query: str) -> dict:
        return run_pipeline(query, use_cache=False)

    background_tasks.add_task(run_evaluation, _pipeline_fn)
    return {"message": "Evaluation started. Check logs for results."}
