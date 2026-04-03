"""
Day 18 — Multi-Query Retrieval

Demonstrates how generating multiple query variations improves retrieval recall.
Uses sentence-transformers for real semantic search over a sample knowledge base.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Sample knowledge base
# ---------------------------------------------------------------------------
knowledge_base = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "source": "rag_intro.txt"},
    {"text": "Vector databases store embeddings and enable fast nearest-neighbor search.", "source": "vector_db.txt"},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "source": "embeddings.txt"},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "source": "chunking.txt"},
    {"text": "Reranking reorders retrieved documents by relevance score before passing to the LLM.", "source": "reranking.txt"},
    {"text": "Hallucinations occur when LLMs generate confident but factually incorrect answers.", "source": "hallucinations.txt"},
    {"text": "Fine-tuning adjusts model weights to change behavior, not to inject new knowledge.", "source": "finetuning.txt"},
    {"text": "Metadata filtering narrows the search space before similarity search runs.", "source": "metadata.txt"},
    {"text": "Hybrid search combines keyword search and semantic search for better recall.", "source": "hybrid.txt"},
    {"text": "Context construction organizes retrieved chunks into a clean, token-aware prompt block.", "source": "context.txt"},
]

# ---------------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# Pre-compute embeddings for the knowledge base
for doc in knowledge_base:
    doc["embedding"] = model.encode(doc["text"])


# ---------------------------------------------------------------------------
# Core search utility
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """Embed query and return top_k most similar documents."""
    query_vec = model.encode(query)
    scored = []
    for doc in docs:
        score = cosine_similarity(query_vec, doc["embedding"])
        scored.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": round(score, 4),
            "matched_query": query
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Multi-query: rule-based variation generator
# (In production, replace this with an LLM call)
# ---------------------------------------------------------------------------

def generate_query_variations(query: str) -> list[str]:
    """
    Generate multiple variations of the original query.
    Covers different phrasings, synonyms, and angles.

    In production: pass the query to an LLM and ask it to generate
    3-5 alternative phrasings. This rule-based version demonstrates
    the concept without requiring an LLM API key.
    """
    variations = [query]  # always include the original

    # Synonym substitutions
    synonym_map = {
        "fix": "resolve",
        "error": "issue",
        "how does": "explain",
        "what is": "define",
        "improve": "enhance",
        "reduce": "minimize",
        "retrieve": "fetch",
        "search": "find",
        "store": "save",
        "fast": "efficient",
    }

    variant = query.lower()
    for original, replacement in synonym_map.items():
        if original in variant:
            variations.append(variant.replace(original, replacement))

    # Perspective shifts
    if "how" in query.lower():
        variations.append(query.lower().replace("how", "what is the process for"))
    if "why" in query.lower():
        variations.append(query.lower().replace("why", "what is the reason"))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique


# ---------------------------------------------------------------------------
# Multi-query retrieval pipeline
# ---------------------------------------------------------------------------

def multi_query_retrieve(
    query: str,
    docs: list[dict],
    top_k_per_query: int = 3,
    final_top_n: int = 5
) -> list[dict]:
    """
    Full multi-query retrieval pipeline:
    1. Generate query variations
    2. Run retrieval for each variation
    3. Merge all results
    4. Deduplicate by source
    5. Re-rank merged results by best score
    6. Return top N
    """
    variations = generate_query_variations(query)
    print(f"Original query: '{query}'")
    print(f"Generated {len(variations)} query variations:")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")
    print()

    # Retrieve for each variation
    all_results = []
    for variation in variations:
        results = search(variation, docs, top_k=top_k_per_query)
        all_results.extend(results)

    print(f"Total raw results (before dedup): {len(all_results)}")

    # Deduplicate — keep highest score per unique source
    best_per_source = {}
    for result in all_results:
        source = result["source"]
        if source not in best_per_source or result["score"] > best_per_source[source]["score"]:
            best_per_source[source] = result

    deduped = list(best_per_source.values())
    print(f"After deduplication: {len(deduped)} unique results")

    # Re-rank by score
    deduped.sort(key=lambda x: x["score"], reverse=True)

    return deduped[:final_top_n]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Query Retrieval Demo")
    print("=" * 60 + "\n")

    # Test 1: Query that benefits from variations
    query = "how does retrieval improve search results?"
    results = multi_query_retrieve(query, knowledge_base, top_k_per_query=3, final_top_n=5)

    print("\nFinal merged results:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. Score: {r['score']} | Matched via: '{r['matched_query']}'")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['source']}")

    print("\n" + "=" * 60 + "\n")

    # Test 2: Compare single query vs multi-query
    print("Single query results (for comparison):")
    single_results = search(query, knowledge_base, top_k=3)
    for i, r in enumerate(single_results, 1):
        print(f"\n  {i}. Score: {r['score']}")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['source']}")
