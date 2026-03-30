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

# Load embedding model (downloads on first run, ~80MB)
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
