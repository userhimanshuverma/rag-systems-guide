import re


# ---------------------------------------------------------------------------
# Sample retrieved docs (output from Day 15 query pipeline)
# Already sorted by relevance score descending
# ---------------------------------------------------------------------------
retrieved_docs = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "metadata": {"source": "rag_intro.txt"}, "score": 0.91},
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "metadata": {"source": "rag_intro_v2.txt"}, "score": 0.90},  # duplicate content
    {"text": "Vector databases enable fast semantic search over high-dimensional embeddings.", "metadata": {"source": "vector_db.txt"}, "score": 0.85},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "metadata": {"source": "embeddings.txt"}, "score": 0.78},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "metadata": {"source": "chunking.txt"}, "score": 0.72},
    {"text": "Reranking reorders retrieved documents by relevance before passing context to the LLM.", "metadata": {"source": "reranking.txt"}, "score": 0.65},
]


# ---------------------------------------------------------------------------
# Step 1 — Clean individual chunk text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove stray characters.
    Keeps the content clean before assembling into context.
    """
    text = re.sub(r'\s+', ' ', text)   # collapse multiple spaces/newlines
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Step 2 — Deduplicate chunks
# ---------------------------------------------------------------------------

def deduplicate(docs: list[dict]) -> list[dict]:
    """
    Remove chunks with identical text content.
    Keeps the first occurrence (highest relevance score since list is pre-sorted).
    """
    seen = set()
    unique = []
    for doc in docs:
        normalized = doc["text"].strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(doc)
    return unique


# ---------------------------------------------------------------------------
# Step 3 — Order by relevance score
# ---------------------------------------------------------------------------

def order_by_relevance(docs: list[dict]) -> list[dict]:
    """
    Sort chunks by score descending — most relevant first.
    The LLM pays more attention to content at the start of the context window.
    """
    return sorted(docs, key=lambda x: x.get("score", 0), reverse=True)


# ---------------------------------------------------------------------------
# Step 4 — Build the context block with token budget
# ---------------------------------------------------------------------------

def build_context(docs: list[dict], max_chars: int = 1500) -> str:
    """
    Assemble retrieved chunks into a formatted context string.

    - Respects a character budget (proxy for token limit)
    - Includes source metadata for traceability
    - Stops adding chunks once budget is reached
    """
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = clean_text(doc["text"])
        source = doc.get("metadata", {}).get("source", "unknown")
        score = doc.get("score", "N/A")

        # Format each chunk with its source label
        chunk_block = f"[{i}] Source: {source} (relevance: {score})\n{text}"
        chunk_len = len(chunk_block)

        # Stop if adding this chunk would exceed the budget
        if total_chars + chunk_len > max_chars:
            break

        context_parts.append(chunk_block)
        total_chars += chunk_len

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Step 5 — Full context construction pipeline
# ---------------------------------------------------------------------------

def construct_context(docs: list[dict], max_chars: int = 1500) -> str:
    """
    Full pipeline:
    Clean → Deduplicate → Order by Relevance → Build Context Block
    """
    # Clean each chunk
    for doc in docs:
        doc["text"] = clean_text(doc["text"])

    # Remove duplicates
    docs = deduplicate(docs)
    print(f"After deduplication: {len(docs)} chunks")

    # Order by relevance
    docs = order_by_relevance(docs)
    print(f"Ordered by relevance score")

    # Build the context string
    context = build_context(docs, max_chars=max_chars)
    print(f"Context built: {len(context)} characters\n")

    return context


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Context Construction Pipeline ===\n")

    context = construct_context(retrieved_docs, max_chars=1500)

    print("--- Final Context Block ---\n")
    print(context)

    print("\n--- How this gets used in a prompt ---\n")
    query = "How does RAG improve LLM answers?"
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    print(prompt)
