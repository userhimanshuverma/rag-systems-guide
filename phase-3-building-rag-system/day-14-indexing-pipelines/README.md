# Day 14 — Indexing Pipelines

> "Prepared data is useless until it's searchable. Indexing is what makes it searchable."

---

## The Problem

Yesterday you built an ingestion pipeline. Your documents are loaded, cleaned, chunked, and tagged with metadata. You have a clean list of structured chunks ready to go.

But they're still just text sitting in memory. You can't search them. You can't retrieve them by meaning. They're prepared — but not indexed.

Indexing is the step that transforms those chunks into something a retrieval system can actually search. It's where text becomes vectors, and vectors get stored in a way that enables fast, meaning-based lookup.

This is the moment raw data becomes searchable intelligence.

---

## What Indexing Actually Is

Indexing has two jobs:

1. **Embed** — convert each chunk's text into a vector using an embedding model
2. **Store** — save that vector alongside the original text and metadata

After indexing, every chunk in your knowledge base has a vector representation. When a query comes in, it gets embedded the same way — and the system finds the stored vectors closest to it.

> Indexing = Embedding + Storage

That's it. But getting it right is what makes retrieval fast and accurate.

---

## The Library Catalog Analogy

Think of a library before and after cataloging.

Before cataloging: thousands of books stacked randomly. You'd have to read every book to find what you need.

After cataloging: every book has a card in the catalog — title, author, subject, location. You search the catalog, find the card, go straight to the book.

Indexing does the same thing for your document chunks. The embedding is the catalog card — a compact representation that captures what the chunk is about. The vector database is the catalog itself — organized for fast lookup.

Without indexing, retrieval means scanning every chunk. With indexing, retrieval means finding the nearest vectors in milliseconds.

---

## How It Works Step by Step

**Step 1 — Take processed chunks from ingestion**
Each chunk has text and metadata. This is the output from Day 13.

**Step 2 — Pass text through an embedding model**
The embedding model reads each chunk and outputs a vector — a list of numbers representing the chunk's meaning. All chunks are embedded using the same model so their vectors are comparable.

**Step 3 — Store vector + text + metadata together**
Each index entry contains three things: the embedding (for search), the original text (to return to the LLM), and the metadata (for filtering and tracing).

**Step 4 — Index is ready for querying**
At query time, the user's question gets embedded with the same model. The system finds the stored vectors closest to the query vector and returns the associated text.

---

## Architecture Flow

```
Processed Chunks (from Day 13)
        ↓
  Embedding Model
(converts text → vector)
        ↓
  Vector + Text + Metadata
        ↓
  Vector Index / Database
(stored, organized for fast search)
        ↓
  Ready for Query Pipeline (Day 15)
```

---

## The Code

Install the dependency first:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Sample processed chunks (output from Day 13 ingestion pipeline)
documents = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents.", "metadata": {"source": "rag_intro.txt", "chunk_index": 0}},
    {"text": "Vector databases enable fast semantic search over embeddings.", "metadata": {"source": "vector_db.txt", "chunk_index": 0}},
    {"text": "Embeddings convert text into numerical vectors that capture meaning.", "metadata": {"source": "embeddings.txt", "chunk_index": 0}},
    {"text": "Chunking splits large documents into smaller pieces for better retrieval.", "metadata": {"source": "chunking.txt", "chunk_index": 0}},
    {"text": "Reranking reorders retrieved documents by relevance before passing to LLM.", "metadata": {"source": "reranking.txt", "chunk_index": 0}},
]

# Load embedding model (downloads on first run, ~90MB)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")


def create_embeddings(docs: list[dict]) -> np.ndarray:
    """
    Convert document texts into embedding vectors.
    Returns a numpy array of shape (num_docs, embedding_dim).
    """
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings


def build_index(docs: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Combine each document's text, metadata, and embedding into a single index entry.
    This is what gets stored in a vector database.
    """
    index = []
    for doc, emb in zip(docs, embeddings):
        index.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "embedding": emb.tolist()  # convert numpy array to list for storage
        })
    return index


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a score between -1 and 1. Higher = more similar.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def search_index(query: str, index: list[dict], top_k: int = 3) -> list[dict]:
    """
    Search the index for the most relevant documents to a query.
    Embeds the query, computes similarity against all stored vectors,
    and returns the top_k most similar results.
    """
    # Embed the query using the same model
    query_embedding = model.encode(query)

    # Score every document in the index
    scored = []
    for entry in index:
        score = cosine_similarity(query_embedding, np.array(entry["embedding"]))
        scored.append({
            "text": entry["text"],
            "metadata": entry["metadata"],
            "score": round(float(score), 4)
        })

    # Sort by score descending and return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # Step 1: Create embeddings for all documents
    print("Creating embeddings...")
    embeddings = create_embeddings(documents)
    print(f"Embedding shape: {embeddings.shape}")  # (num_docs, 384)

    # Step 2: Build the index
    vector_index = build_index(documents, embeddings)
    print(f"Indexed {len(vector_index)} documents\n")

    # Step 3: Preview one index entry (without the full embedding vector)
    sample = vector_index[0].copy()
    sample["embedding"] = f"[{len(vector_index[0]['embedding'])} dimensions]"
    print("Sample index entry:")
    print(json.dumps(sample, indent=2))

    # Step 4: Run a test search
    query = "How does semantic search work?"
    print(f"\nQuery: '{query}'")
    print("Top results:")
    results = search_index(query, vector_index, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Score: {result['score']}")
        print(f"     Text: {result['text']}")
        print(f"     Source: {result['metadata']['source']}")
```

---

## Expected Output

```
Loading embedding model...
Model loaded.

Creating embeddings...
Embedding shape: (5, 384)
Indexed 5 documents

Sample index entry:
{
  "text": "RAG improves LLM accuracy by retrieving relevant documents.",
  "metadata": {
    "source": "rag_intro.txt",
    "chunk_index": 0
  },
  "embedding": "[384 dimensions]"
}

Query: 'How does semantic search work?'
Top results:

  1. Score: 0.5073
     Text: Vector databases enable fast semantic search over embeddings.
     Source: vector_db.txt

  2. Score: 0.3345
     Text: Chunking splits large documents into smaller pieces for better retrieval.
     Source: chunking.txt

  3. Score: 0.3058
     Text: Reranking reorders retrieved documents by relevance before passing to LLM.
     Source: reranking.txt
```

The query "How does semantic search work?" correctly surfaces the vector databases chunk as the top result — because it's the most semantically relevant. The scores reflect genuine meaning similarity, not keyword overlap.

---

## What Each Part Does

**`SentenceTransformer("all-MiniLM-L6-v2")`**
Loads a lightweight but capable embedding model. It converts any text into a 384-dimensional vector. The same model must be used for both indexing and querying — otherwise the vectors aren't comparable.

**`create_embeddings(docs)`**
Extracts the text from each document and passes them all to the model at once (batch processing). Returns a numpy array of shape `(num_docs, 384)` — one 384-dimensional vector per document.

**`build_index(docs, embeddings)`**
Pairs each document's text and metadata with its embedding. This is the complete index entry — everything you need to search and retrieve. In a real system, this gets stored in a vector database instead of a Python list.

**`cosine_similarity(vec_a, vec_b)`**
Measures how similar two vectors are by the angle between them. Returns a value between -1 and 1. In practice, text embeddings score between 0 and 1 — higher means more similar.

**`search_index(query, index, top_k)`**
The retrieval function. Embeds the query, scores every indexed document against it, sorts by score, and returns the top K results. This is the core of what a vector database does — just implemented simply here for learning purposes.

---

## Why Embeddings Are Stored With Metadata

The embedding alone isn't enough. When retrieval finds a relevant vector, you need two things:

- The **original text** — to pass to the LLM as context
- The **metadata** — to know where it came from, filter by source, and trace answers back to documents

Storing all three together (embedding + text + metadata) means a single lookup gives you everything you need for the next step.

---

## Moving to a Real Vector Database

This implementation stores everything in a Python list — fine for learning, not for production.

In a real system, you'd replace the list with a vector database that:
- Persists data to disk (survives restarts)
- Scales to millions of vectors
- Uses optimized indexing for fast nearest-neighbor search
- Supports metadata filtering alongside similarity search

The logic stays identical — embed, store, search. Only the storage layer changes.

---

## Key Takeaways

- Indexing converts chunks into embeddings and stores them for fast retrieval
- The same embedding model must be used for both indexing and querying
- Each index entry stores three things: embedding (for search), text (for LLM), metadata (for filtering)
- `all-MiniLM-L6-v2` produces 384-dimensional vectors — compact, fast, and good for general use
- Cosine similarity measures the angle between vectors — higher score means more semantically similar
- This in-memory implementation is great for learning; production systems use a dedicated vector database
- The indexing pipeline is a one-time (or periodic) operation — querying happens in real time

---

## What's Next

Your index is built. Documents are embedded and stored. The system is ready to be searched.

On **Day 15**, we'll build the Query Pipeline — what happens from the moment a user types a question to the moment relevant chunks are retrieved. We'll connect the embedding model, the index, and the retrieval logic into a complete, working search flow.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
