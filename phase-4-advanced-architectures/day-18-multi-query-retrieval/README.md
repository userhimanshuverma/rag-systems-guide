# Day 18 — Multi-Query Retrieval

> "One question asked one way will always miss something. Ask it five ways and you find everything."

---

## The Problem

You have a well-built RAG pipeline. Embeddings, vector database, similarity search — all working correctly.

But a user asks: *"How does retrieval improve search results?"*

Your retriever returns 3 chunks. They're relevant. But there's a fourth chunk — about chunking strategies — that's also directly relevant. It just uses different vocabulary. The single query vector didn't land close enough to it in the embedding space.

The user gets an incomplete answer. Not because the data wasn't there. Because the query only captured one angle of the question.

This is the recall problem. And it's one of the most common failure modes in production RAG systems.

---

## The Simple Fix

> Multi-query retrieval = run the same question multiple ways, merge the results.

Instead of sending one query to the retriever, you generate several variations of the same question — different phrasings, synonyms, perspectives — and retrieve for each one. Then you merge all the results, deduplicate, and rerank.

Each variation casts a slightly different net. Together, they catch what a single query would miss.

---

## The Google Analogy

You've done this yourself without realizing it.

You search for something on Google. The results aren't quite right. So you rephrase — try different words, a different angle. Suddenly the right result appears.

You were doing multi-query retrieval manually. The difference is that in a RAG system, you automate this process. Instead of waiting for the user to rephrase, the system generates variations upfront and searches with all of them simultaneously.

---

## Step-by-Step Flow

**Step 1 — Original query arrives**
User asks: *"How does retrieval improve search results?"*

**Step 2 — Generate query variations**
The system produces multiple phrasings of the same question:
- "How does retrieval improve search results?" (original)
- "Explain retrieval improve search results?"
- "How does retrieval enhance search results?"
- "How does retrieval improve find results?"

**Step 3 — Retrieve for each variation**
Each variation is embedded and searched independently. Each returns its own top K results.

**Step 4 — Merge all results**
All results from all queries are combined into one pool.

**Step 5 — Deduplicate**
Remove duplicate documents. Keep the highest score for each unique source.

**Step 6 — Rerank and return top N**
Sort the merged, deduplicated pool by score. Return the top N for context construction.

---

## Architecture Flow

```
User Query
    ↓
Query Variation Generator
(rule-based or LLM-powered)
    ↓
Multiple Query Variations
[q1, q2, q3, q4, q5]
    ↓
Retrieval (per query)
[results1] [results2] [results3] [results4] [results5]
    ↓
Merge All Results
    ↓
Deduplicate (keep best score per source)
    ↓
Re-rank by Score
    ↓
Top N Results → Context Construction → LLM
```

---

## The Code

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Sample knowledge base
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

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

for doc in knowledge_base:
    doc["embedding"] = model.encode(doc["text"])


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


def generate_query_variations(query: str) -> list[str]:
    """
    Generate multiple variations of the original query.
    Rule-based version — in production, replace with an LLM call.
    """
    variations = [query]

    synonym_map = {
        "fix": "resolve", "error": "issue",
        "how does": "explain", "what is": "define",
        "improve": "enhance", "reduce": "minimize",
        "retrieve": "fetch", "search": "find",
        "store": "save", "fast": "efficient",
    }

    variant = query.lower()
    for original, replacement in synonym_map.items():
        if original in variant:
            variations.append(variant.replace(original, replacement))

    if "how" in query.lower():
        variations.append(query.lower().replace("how", "what is the process for"))
    if "why" in query.lower():
        variations.append(query.lower().replace("why", "what is the reason"))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def multi_query_retrieve(
    query: str,
    docs: list[dict],
    top_k_per_query: int = 3,
    final_top_n: int = 5
) -> list[dict]:
    """
    Full multi-query retrieval pipeline:
    Generate variations → retrieve per query → merge → deduplicate → rerank
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

    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped[:final_top_n]


if __name__ == "__main__":
    query = "how does retrieval improve search results?"
    results = multi_query_retrieve(query, knowledge_base, top_k_per_query=3, final_top_n=5)

    print("\nFinal merged results:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. Score: {r['score']} | Matched via: '{r['matched_query']}'")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['source']}")

    print("\n--- Single query results (for comparison) ---")
    single = search(query, knowledge_base, top_k=3)
    for i, r in enumerate(single, 1):
        print(f"\n  {i}. Score: {r['score']}")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['source']}")
```

---

## Expected Output

```
Original query: 'how does retrieval improve search results?'
Generated 5 query variations:
  1. how does retrieval improve search results?
  2. explain retrieval improve search results?
  3. how does retrieval enhance search results?
  4. how does retrieval improve find results?
  5. what is the process for does retrieval improve search results?

Total raw results (before dedup): 15
After deduplication: 4 unique results

Final merged results:

  1. Score: 0.5797 | Matched via: 'how does retrieval improve search results?'
     Text: Hybrid search combines keyword search and semantic search for better recall.

  2. Score: 0.4997 | Matched via: 'how does retrieval improve find results?'
     Text: Metadata filtering narrows the search space before similarity search runs.

  3. Score: 0.4651 | Matched via: 'how does retrieval improve find results?'
     Text: Chunking splits large documents into smaller pieces for more precise retrieval.

  4. Score: 0.4553 | Matched via: 'how does retrieval improve search results?'
     Text: Reranking reorders retrieved documents by relevance score before passing to the LLM.

--- Single query results (for comparison) ---

  1. Score: 0.5797 — Hybrid search...
  2. Score: 0.4992 — Metadata filtering...
  3. Score: 0.4553 — Reranking...
```

Multi-query retrieved 4 unique results. Single query only got 3. The `chunking.txt` chunk was surfaced by the "find" variation — it wouldn't have appeared with the original query alone.

---

## What Each Part Does

**`generate_query_variations(query)`**
Produces multiple phrasings of the original query using synonym substitution and perspective shifts. This is a rule-based implementation for demonstration. In production, you'd pass the query to an LLM and ask it to generate 3-5 alternative phrasings — that produces much richer variations.

**`search(query, docs, top_k)`**
Standard semantic search — embeds the query and returns top K results by cosine similarity. Called once per variation.

**`multi_query_retrieve(...)`**
The orchestrator. Generates variations, runs search for each, merges all results, deduplicates by source (keeping the best score), re-sorts, and returns the top N. The `matched_query` field tells you which variation found each result — useful for debugging.

**Deduplication by source**
When multiple query variations retrieve the same document, we keep only the highest-scoring occurrence. This prevents the same chunk from appearing multiple times in the final context.

---

## Rule-Based vs LLM-Based Query Generation

This implementation uses simple synonym substitution. It works for demonstration but has limits.

In production, the better approach is to use an LLM to generate variations:

```python
# Conceptual — requires an LLM API
def generate_queries_with_llm(query: str, llm) -> list[str]:
    prompt = f"""Generate 4 different phrasings of this question for a search system.
Each phrasing should capture the same intent but use different words or angles.
Return only the questions, one per line.

Original question: {query}"""
    response = llm(prompt)
    variations = [query] + response.strip().split("\n")
    return variations
```

LLM-generated variations are more semantically diverse and handle complex, ambiguous queries much better than rule-based substitution.

---

## Why Multi-Query Matters

**Vocabulary mismatch**
Users phrase questions differently than documents are written. Multiple variations increase the chance of hitting the right vocabulary.

**Ambiguous queries**
A single query might be interpreted one way by the embedding model. Variations force different interpretations, catching more relevant content.

**Recall vs precision**
Multi-query improves recall — you find more relevant documents. Deduplication and reranking then restore precision — you still only pass the best ones to the LLM.

**Low cost, high impact**
Generating 3-5 query variations and running parallel searches is cheap. The improvement in retrieval coverage is often significant, especially for complex or ambiguous questions.

---

## Key Takeaways

- A single query only captures one angle of a question — multi-query covers more ground
- Generate 3-5 variations of the original query, retrieve for each, then merge results
- Deduplication keeps the best score per unique source — prevents the same chunk appearing twice
- The `matched_query` field shows which variation found each result — invaluable for debugging
- Rule-based variation works for demos; LLM-generated variations work better in production
- Multi-query improves recall without hurting precision — deduplication + reranking handles the rest
- Retrieval quality is not just about data — it's also about how you ask

---

## What's Next

Multi-query retrieval improves recall by asking the same question multiple ways.

On **Day 19**, we go further — Agentic RAG, where an AI agent dynamically decides *what* to retrieve, *when* to retrieve it, and *whether* to retrieve at all. This is where RAG systems start to feel truly intelligent.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
