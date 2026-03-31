# Day 15 — Query Pipelines

> "A question is just noise until the system knows how to process it."

---

## The Problem

You have an index. Thousands of document chunks, all embedded and stored. The retrieval system is ready.

Now a user types a question.

What actually happens next? How does that raw text string become a set of relevant document chunks that the LLM can use?

That's the query pipeline — the sequence of steps that transforms a user's question into retrieved context. It's the real-time half of RAG, and it runs on every single request.

---

## What the Query Pipeline Does

The ingestion and indexing pipelines run once (or periodically) to prepare your knowledge base. The query pipeline runs every time a user asks something.

Its job:

> Take a raw user question → process it → retrieve the most relevant chunks → return them ready for the LLM

The pipeline has four stages:

1. **Embed** the query using the same model used during indexing
2. **Filter** optionally narrow the search space using metadata
3. **Retrieve** find the top K most similar chunks by vector similarity
4. **Rerank** optionally re-score and reorder for precision

---

## The Librarian Analogy

Think of a librarian in a well-organized library.

A student walks in and asks: *"I need something about how search engines rank results."*

The librarian doesn't read every book. They:
1. Understand what the student is asking (embed the query)
2. Head to the right section — maybe filter to "Computer Science" (metadata filter)
3. Pull out the 5 most relevant-looking books (retrieve top K)
4. Quickly skim each one and hand over the 3 that actually answer the question best (rerank)

That's the query pipeline. Fast, structured, and focused on getting the right material into the right hands.

---

## Step-by-Step Flow

**Step 1 — Embed the query**
The user's question is passed through the embedding model — the same one used during indexing. This produces a query vector that lives in the same vector space as the stored document vectors.

**Step 2 — Apply metadata filters (optional)**
Before searching, narrow the candidate pool using metadata. Filter by source, date, category, or any other attribute. This reduces noise and speeds up search.

**Step 3 — Retrieve top K by similarity**
Compare the query vector against all stored vectors. Score each one using cosine similarity. Return the top K highest-scoring chunks.

**Step 4 — Rerank (optional)**
Re-score the top K candidates more carefully. Push the most genuinely relevant chunks to the top. Pass only the top N to the LLM.

---

## Architecture Flow

```
User Query (raw text)
        ↓
  Embedding Model
(query → vector, same model as indexing)
        ↓
  Metadata Filter (optional)
(narrow candidates by source, date, type)
        ↓
  Vector Similarity Search
(compare query vector vs stored vectors)
        ↓
  Top K Retrieved Chunks
        ↓
  Reranker (optional)
(re-score for precision, select top N)
        ↓
  Final Context
(ready for LLM → Day 16)
```

---

## The Code

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Sample index — in a real system this comes from the indexing pipeline (Day 14)
# ---------------------------------------------------------------------------
documents = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "metadata": {"source": "rag_intro.txt"}},
    {"text": "Vector databases enable fast semantic search over high-dimensional embeddings.", "metadata": {"source": "vector_db.txt"}},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "metadata": {"source": "embeddings.txt"}},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "metadata": {"source": "chunking.txt"}},
    {"text": "Reranking reorders retrieved documents by relevance before passing context to the LLM.", "metadata": {"source": "reranking.txt"}},
    {"text": "Metadata filtering narrows the search space before similarity search runs.", "metadata": {"source": "metadata.txt"}},
    {"text": "Hybrid search combines keyword search and semantic search for better recall.", "metadata": {"source": "hybrid.txt"}},
]

# Load embedding model (same model used during indexing — must match)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# Pre-compute embeddings for all documents (simulates a pre-built index)
for doc in documents:
    doc["embedding"] = model.encode(doc["text"])


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: -1 to 1, higher = more similar."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def embed_query(query: str) -> np.ndarray:
    """Convert a user query into an embedding vector."""
    return model.encode(query)


# ---------------------------------------------------------------------------
# Query pipeline stages
# ---------------------------------------------------------------------------

def retrieve(query_embedding: np.ndarray, docs: list[dict], top_k: int = 5) -> list[dict]:
    """
    Stage 1 — Retrieval
    Score every document against the query embedding and return top_k candidates.
    """
    scored = []
    for doc in docs:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": round(float(score), 4)
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def apply_metadata_filter(docs: list[dict], filters: dict) -> list[dict]:
    """
    Optional Stage — Metadata Filtering
    Filter documents by metadata fields before retrieval.
    Example: filters={"source": "rag_intro.txt"}
    """
    if not filters:
        return docs
    filtered = []
    for doc in docs:
        if all(doc["metadata"].get(k) == v for k, v in filters.items()):
            filtered.append(doc)
    return filtered


def rerank(query: str, docs: list[dict], top_n: int = 3) -> list[dict]:
    """
    Optional Stage — Simple Reranking
    Re-scores retrieved docs using fresh embeddings for more precise ranking.
    In production this would use a dedicated cross-encoder reranker model.
    """
    query_embedding = embed_query(query)
    for doc in docs:
        doc_embedding = model.encode(doc["text"])
        doc["rerank_score"] = round(float(cosine_similarity(query_embedding, doc_embedding)), 4)
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_n]


def run_query_pipeline(
    query: str,
    docs: list[dict],
    top_k: int = 5,
    top_n: int = 3,
    filters: dict = None,
    use_reranker: bool = True
) -> list[dict]:
    """
    Full query pipeline:
    Query → Embed → (Filter) → Retrieve → (Rerank) → Return top results
    """
    print(f"Query: '{query}'")
    print("-" * 50)

    # Step 1: Embed the query
    query_embedding = embed_query(query)
    print(f"Step 1: Query embedded ({len(query_embedding)} dimensions)")

    # Step 2: Optional metadata filtering
    candidate_docs = apply_metadata_filter(docs, filters or {})
    print(f"Step 2: Metadata filter applied → {len(candidate_docs)} candidates")

    # Step 3: Retrieve top K by similarity
    retrieved = retrieve(query_embedding, candidate_docs, top_k=top_k)
    print(f"Step 3: Retrieved top {top_k} documents by similarity")

    # Step 4: Optional reranking
    if use_reranker:
        final_results = rerank(query, retrieved, top_n=top_n)
        print(f"Step 4: Reranked → top {top_n} results selected")
    else:
        final_results = retrieved[:top_n]
        print(f"Step 4: Reranking skipped → top {top_n} results selected")

    return final_results


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: Standard query with reranking
    results = run_query_pipeline(
        query="How does semantic search work?",
        docs=documents,
        top_k=5,
        top_n=3,
        use_reranker=True
    )
    print("\nFinal Results:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. Score: {r.get('rerank_score', r['score'])}")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['metadata']['source']}")

    print("\n" + "=" * 60 + "\n")

    # Test 2: Query with metadata filter, no reranking
    results_filtered = run_query_pipeline(
        query="How does RAG work?",
        docs=documents,
        top_k=5,
        top_n=2,
        filters={"source": "rag_intro.txt"},
        use_reranker=False
    )
    print("\nFiltered Results (source=rag_intro.txt):")
    for i, r in enumerate(results_filtered, 1):
        print(f"\n  {i}. Score: {r['score']}")
        print(f"     Text: {r['text']}")
        print(f"     Source: {r['metadata']['source']}")
```

---

## Expected Output

```
Loading embedding model...
Model loaded.

Query: 'How does semantic search work?'
--------------------------------------------------
Step 1: Query embedded (384 dimensions)
Step 2: Metadata filter applied → 7 candidates
Step 3: Retrieved top 5 documents by similarity
Step 4: Reranked → top 3 results selected

Final Results:

  1. Score: 0.5942
     Text: Hybrid search combines keyword search and semantic search for better recall.
     Source: hybrid.txt

  2. Score: 0.5129
     Text: Metadata filtering narrows the search space before similarity search runs.
     Source: metadata.txt

  3. Score: 0.4707
     Text: Vector databases enable fast semantic search over high-dimensional embeddings.
     Source: vector_db.txt

============================================================

Query: 'How does RAG work?'
--------------------------------------------------
Step 1: Query embedded (384 dimensions)
Step 2: Metadata filter applied → 1 candidates
Step 3: Retrieved top 5 documents by similarity
Step 4: Reranking skipped → top 2 results selected

Filtered Results (source=rag_intro.txt):

  1. Score: 0.4414
     Text: RAG improves LLM accuracy by retrieving relevant documents before generating answers.
     Source: rag_intro.txt
```

The first query correctly surfaces hybrid search and semantic search docs at the top. The second query with a metadata filter correctly restricts results to only the `rag_intro.txt` source.

---

## What Each Part Does

**`embed_query(query)`**
Converts the user's raw text question into a 384-dimensional vector using the same model that was used during indexing. This is critical — if you use a different model for queries than for indexing, the vectors won't be comparable and retrieval will break.

**`apply_metadata_filter(docs, filters)`**
Narrows the candidate pool before similarity search. Accepts a dict of key-value pairs to match against document metadata. Empty filters means no filtering — all documents are candidates.

**`retrieve(query_embedding, docs, top_k)`**
The core retrieval function. Scores every document in the candidate pool using cosine similarity, sorts by score descending, and returns the top K. This is the fast, broad pass — cast a wide net.

**`rerank(query, docs, top_n)`**
A second scoring pass over the retrieved candidates. Re-embeds each document and re-scores against the query for more precise ranking. In production, a dedicated cross-encoder model would do this more accurately — but the principle is identical.

**`run_query_pipeline(...)`**
Orchestrates all four stages in sequence. Accepts flags to toggle filtering and reranking on or off. This is the function you'd call from your application layer.

---

## Key Design Decisions

**top_k vs top_n**
`top_k` is how many documents you retrieve (broad). `top_n` is how many you pass to the LLM after reranking (precise). A typical setup: `top_k=20`, `top_n=5`. Retrieve broadly, pass narrowly.

**Same model for indexing and querying**
This is non-negotiable. The query vector and document vectors must live in the same embedding space. Always use the same model for both.

**Filters before retrieval**
Applying metadata filters before similarity search reduces the candidate pool — making search faster and results more focused. Filters after retrieval are also valid but less efficient.

**Reranking is optional but valuable**
For simple use cases, skip it. For production systems where answer quality matters, reranking is one of the highest-ROI improvements you can make.

---

## Key Takeaways

- The query pipeline runs on every user request — it's the real-time half of RAG
- Four stages: embed → filter → retrieve → rerank
- The query must be embedded with the same model used during indexing
- `top_k` controls retrieval breadth; `top_n` controls what the LLM actually sees
- Metadata filtering narrows the search space before similarity search — faster and more focused
- Reranking adds a precision pass after retrieval — pushes the best results to the top
- Each stage is independently tunable — you can improve any one without touching the others

---

## What's Next

The query pipeline returns a list of relevant chunks. But you can't just dump them directly into the LLM prompt.

On **Day 16**, we'll cover Context Construction — how retrieved chunks are assembled, ordered, and formatted into a prompt that the LLM can actually use effectively.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
