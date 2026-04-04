"""
Day 19 — Agentic RAG

Demonstrates a decision-loop agent that dynamically decides whether to:
- Retrieve more documents
- Refine the query and retry
- Stop and generate an answer
- Admit it couldn't find enough information

No heavy frameworks — pure Python with sentence-transformers for real semantic search.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
knowledge_base = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "source": "rag_intro.txt"},
    {"text": "Vector databases store embeddings and enable fast nearest-neighbor search at scale.", "source": "vector_db.txt"},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "source": "embeddings.txt"},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "source": "chunking.txt"},
    {"text": "Reranking reorders retrieved documents by relevance score before passing to the LLM.", "source": "reranking.txt"},
    {"text": "Hallucinations occur when LLMs generate confident but factually incorrect answers.", "source": "hallucinations.txt"},
    {"text": "Fine-tuning adjusts model weights to change behavior, not to inject new knowledge.", "source": "finetuning.txt"},
    {"text": "Metadata filtering narrows the search space before similarity search runs.", "source": "metadata.txt"},
    {"text": "Hybrid search combines keyword search and semantic search for better recall.", "source": "hybrid.txt"},
    {"text": "Context construction organizes retrieved chunks into a clean, token-aware prompt block.", "source": "context.txt"},
    {"text": "Agentic RAG uses a decision loop to dynamically decide when and what to retrieve.", "source": "agentic.txt"},
    {"text": "Multi-query retrieval generates multiple query variations to improve recall.", "source": "multi_query.txt"},
]

# ---------------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

for doc in knowledge_base:
    doc["embedding"] = model.encode(doc["text"])


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """Semantic search — returns top_k most relevant documents."""
    query_vec = model.encode(query)
    scored = []
    for doc in docs:
        score = cosine_similarity(query_vec, doc["embedding"])
        scored.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": round(score, 4)
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Agent decision functions
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.45  # minimum score to consider a result "good enough"
COVERAGE_THRESHOLD = 2       # minimum number of good results needed


def evaluate_results(results: list[dict]) -> dict:
    """
    Evaluate whether retrieved results are sufficient to answer the query.

    Returns a decision dict:
    - status: "sufficient" | "insufficient" | "empty"
    - good_count: number of results above threshold
    - best_score: highest relevance score
    """
    if not results:
        return {"status": "empty", "good_count": 0, "best_score": 0.0}

    good_results = [r for r in results if r["score"] >= RELEVANCE_THRESHOLD]
    best_score = results[0]["score"]

    if len(good_results) >= COVERAGE_THRESHOLD:
        return {"status": "sufficient", "good_count": len(good_results), "best_score": best_score}
    else:
        return {"status": "insufficient", "good_count": len(good_results), "best_score": best_score}


def decide_action(evaluation: dict, step: int, max_steps: int) -> str:
    """
    Agent decision logic.

    Returns one of:
    - "answer"  → results are good enough, generate answer
    - "refine"  → results are weak, try a refined query
    - "broaden" → results are empty, try a broader query
    - "stop"    → max steps reached, give up
    """
    if step >= max_steps - 1:
        return "stop"
    if evaluation["status"] == "sufficient":
        return "answer"
    if evaluation["status"] == "empty":
        return "broaden"
    return "refine"


def refine_query(query: str, attempt: int) -> str:
    """
    Refine the query to improve retrieval on the next attempt.
    Each attempt applies a different refinement strategy.
    """
    strategies = [
        lambda q: q + " explanation and examples",
        lambda q: "detailed overview of " + q,
        lambda q: "how does " + q + " work in practice",
    ]
    strategy = strategies[attempt % len(strategies)]
    return strategy(query)


def broaden_query(query: str) -> str:
    """Broaden a query that returned no results."""
    words = query.split()
    # Keep only the most important words (first 3)
    return " ".join(words[:3]) if len(words) > 3 else query + " overview"


def generate_answer(query: str, results: list[dict]) -> str:
    """
    Simulate answer generation from retrieved context.
    In production, this passes the context to an LLM.
    """
    context_lines = [f"[{i+1}] {r['text']} (source: {r['source']})"
                     for i, r in enumerate(results)]
    context = "\n".join(context_lines)
    return (
        f"Based on the retrieved context, here is the answer to: '{query}'\n\n"
        f"Context used:\n{context}\n\n"
        f"[In production: this context + question would be sent to an LLM for final generation]"
    )


# ---------------------------------------------------------------------------
# Agentic RAG loop
# ---------------------------------------------------------------------------

def agentic_rag(query: str, max_steps: int = 4) -> str:
    """
    Agentic RAG decision loop.

    At each step the agent:
    1. Retrieves documents for the current query
    2. Evaluates whether results are sufficient
    3. Decides: answer / refine / broaden / stop
    4. Iterates if needed
    """
    print(f"{'='*60}")
    print(f"  Agentic RAG — Query: '{query}'")
    print(f"{'='*60}\n")

    current_query = query
    all_results = []

    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        print(f"Query: '{current_query}'")

        # Retrieve
        results = retrieve(current_query, knowledge_base, top_k=3)
        print(f"Retrieved {len(results)} documents:")
        for r in results:
            print(f"  [{r['score']}] {r['text'][:70]}...")

        # Accumulate unique results across steps
        existing_sources = {r["source"] for r in all_results}
        new_results = [r for r in results if r["source"] not in existing_sources]
        all_results.extend(new_results)
        print(f"Total unique results so far: {len(all_results)}")

        # Evaluate
        evaluation = evaluate_results(results)
        print(f"Evaluation: {evaluation}")

        # Decide
        action = decide_action(evaluation, step, max_steps)
        print(f"Agent decision: {action.upper()}\n")

        if action == "answer":
            print("Sufficient context found. Generating answer...\n")
            return generate_answer(query, all_results[:5])

        elif action == "refine":
            current_query = refine_query(current_query, step)
            print(f"Refining query to: '{current_query}'\n")

        elif action == "broaden":
            current_query = broaden_query(current_query)
            print(f"Broadening query to: '{current_query}'\n")

        elif action == "stop":
            break

    # Max steps reached — use whatever we have
    if all_results:
        print("Max steps reached. Using best available results...\n")
        return generate_answer(query, all_results[:5])

    return "Agent could not find sufficient information to answer this query."


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: Query that resolves quickly
    answer = agentic_rag("how does RAG work?", max_steps=4)
    print("\nFINAL ANSWER:")
    print(answer)

    print("\n" + "=" * 60 + "\n")

    # Test 2: Vague query that needs refinement
    answer2 = agentic_rag("search systems", max_steps=4)
    print("\nFINAL ANSWER:")
    print(answer2)
