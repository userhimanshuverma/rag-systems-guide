"""
Pipeline test — runs all components except LLM (Ollama not required).
Tests: retriever, reranker, context builder, prompt, cache, evaluator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time

print("=" * 60)
print("  Production RAG — Component Tests")
print("=" * 60)

# ── 1. Retriever ──────────────────────────────────────────────
print("\n[1] Testing Retriever...")
from app.retriever import Retriever
retriever = Retriever()
assert retriever.index is not None, "FAIL: Index not loaded"
assert retriever.index.ntotal == 13, f"FAIL: Expected 13 vectors, got {retriever.index.ntotal}"

results = retriever.retrieve("What is RAG?", top_k=5)
assert len(results) > 0, "FAIL: No results returned"
assert "text" in results[0], "FAIL: Missing text field"
assert "score" in results[0], "FAIL: Missing score field"
assert "source" in results[0], "FAIL: Missing source field"
print(f"  PASS: Retrieved {len(results)} chunks | top score={results[0]['score']}")
print(f"  Top chunk: {results[0]['text'][:80]}...")

# ── 2. Multi-query Retriever ──────────────────────────────────
print("\n[2] Testing Multi-Query Retriever...")
queries = ["What is RAG?", "explain retrieval augmented generation", "how does RAG work"]
multi_results = retriever.multi_query_retrieve(queries, top_k=5)
assert len(multi_results) >= len(results), "FAIL: Multi-query should return >= single query results"
print(f"  PASS: Multi-query returned {len(multi_results)} unique chunks (vs {len(results)} single)")

# ── 3. Reranker ───────────────────────────────────────────────
print("\n[3] Testing Reranker...")
from app.reranker import rerank
reranked = rerank("What is RAG?", results, top_n=3)
assert len(reranked) <= 3, "FAIL: Reranker returned more than top_n"
assert "rerank_score" in reranked[0], "FAIL: Missing rerank_score"
assert reranked[0]["rerank_score"] >= reranked[-1]["rerank_score"], "FAIL: Not sorted by score"
print(f"  PASS: Reranked to {len(reranked)} chunks | top rerank_score={reranked[0]['rerank_score']}")

# ── 4. Context Builder ────────────────────────────────────────
print("\n[4] Testing Context Builder...")
from app.context_builder import build_context
context = build_context(reranked, max_chars=2000)
assert len(context) > 0, "FAIL: Empty context"
assert "Source:" in context, "FAIL: Missing source labels"
assert len(context) <= 2000, f"FAIL: Context exceeds budget ({len(context)} chars)"
print(f"  PASS: Context built | {len(context)} chars")
print(f"  Preview: {context[:120]}...")

# ── 5. Prompt Builder ─────────────────────────────────────────
print("\n[5] Testing Prompt Builder...")
from app.prompt import build_rag_prompt
prompt = build_rag_prompt(context, "What is RAG?")
assert "ONLY" in prompt, "FAIL: Missing grounding instruction"
assert "CONTEXT" in prompt, "FAIL: Missing context section"
assert "QUESTION" in prompt, "FAIL: Missing question section"
assert context in prompt, "FAIL: Context not embedded in prompt"
print(f"  PASS: Prompt built | {len(prompt)} chars")

# ── 6. Cache ──────────────────────────────────────────────────
print("\n[6] Testing Cache...")
from app import cache
key = cache.make_key("test query", "test context")
assert len(key) == 64, "FAIL: Cache key should be 64-char SHA-256"

cache.set(key, "test response")
retrieved = cache.get(key)
assert retrieved == "test response", "FAIL: Cache get returned wrong value"

stats = cache.stats()
assert stats["total_entries"] >= 1, "FAIL: Cache stats show 0 entries"

cleared = cache.clear()
assert cleared >= 1, "FAIL: Clear returned 0"
assert cache.get(key) is None, "FAIL: Cache not cleared"
print(f"  PASS: Cache set/get/clear/stats all working")

# ── 7. Keyword filter (hybrid-style) ─────────────────────────
print("\n[7] Testing Keyword Filter...")
filtered = retriever.retrieve("retrieval", top_k=10, keyword_filter="embedding")
for chunk in filtered:
    assert "embedding" in chunk["text"].lower(), f"FAIL: Keyword filter missed: {chunk['text'][:60]}"
print(f"  PASS: Keyword filter returned {len(filtered)} chunks containing 'embedding'")

# ── 8. Evaluator (without LLM — uses mock pipeline) ──────────
print("\n[8] Testing Evaluator...")
from app.evaluator import run_evaluation

def mock_pipeline(query: str) -> dict:
    """Mock pipeline that returns retrieval results without LLM."""
    chunks = retriever.retrieve(query, top_k=5)
    reranked_chunks = rerank(query, chunks, top_n=3)
    return {
        "answer": " ".join([c["text"] for c in reranked_chunks]),
        "chunks": reranked_chunks,
        "latency": 0.1
    }

eval_results = run_evaluation(mock_pipeline)
assert "avg_answer_recall" in eval_results, "FAIL: Missing avg_answer_recall"
assert "source_hit_rate" in eval_results, "FAIL: Missing source_hit_rate"
assert eval_results["total_cases"] == 4, f"FAIL: Expected 4 test cases, got {eval_results['total_cases']}"
assert eval_results["source_hit_rate"] > 0, "FAIL: Source hit rate is 0"
print(f"  PASS: Evaluation complete")
print(f"    avg_answer_recall : {eval_results['avg_answer_recall']}")
print(f"    source_hit_rate   : {eval_results['source_hit_rate']}")
print(f"    avg_latency_s     : {eval_results['avg_latency_s']}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("  LLM (Ollama/Mistral) not tested — requires `ollama serve`")
print("  Run: ollama pull mistral && ollama serve")
print("  Then: uvicorn app.main:app --reload --port 8000")
print("=" * 60)
