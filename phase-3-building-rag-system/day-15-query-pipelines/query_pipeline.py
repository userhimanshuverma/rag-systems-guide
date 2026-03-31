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

# ---------------------------------------------------------------------------
# Load embedding model (same model used during indexing — must match)
# ---------------------------------------------------------------------------
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
    Filter documents by metadata fields before or after retrieval.
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
    Here we re-embed and re-score as a simple demonstration.
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

    # Step 2: Optional metadata filtering (pre-retrieval)
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
    # Test 1: Standard query
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

    # Test 2: Query with metadata filter
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
