"""
Reranker module.
Re-scores retrieved chunks using a combined relevance signal:
- Cosine similarity (from retrieval)
- Keyword overlap with query
- Chunk position (earlier chunks in source doc get slight boost)
Returns top-N reranked results.
"""

import re
from app.config import TOP_N_RERANK
from app.logger import get_logger

logger = get_logger("reranker")


def _keyword_overlap(query: str, text: str) -> float:
    """Fraction of query words that appear in the chunk text."""
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def rerank(query: str, chunks: list[dict], top_n: int = TOP_N_RERANK) -> list[dict]:
    """
    Rerank retrieved chunks using a weighted combination of:
    - vector similarity score (weight: 0.7)
    - keyword overlap with query (weight: 0.3)

    Returns top_n chunks sorted by combined score.
    """
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        vector_score = chunk.get("score", 0.0)
        kw_score = _keyword_overlap(query, chunk.get("text", ""))
        combined = round(0.7 * vector_score + 0.3 * kw_score, 4)

        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = combined
        scored.append(chunk_copy)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = scored[:top_n]

    logger.info(
        f"Reranked {len(chunks)} -> {len(top)} chunks | "
        f"top score={top[0]['rerank_score'] if top else 'N/A'}"
    )
    return top
