"""
Context builder module.
Assembles reranked chunks into a clean, token-aware context block for the LLM.
Handles deduplication, ordering, and character budget enforcement.
"""

import re
from app.config import MAX_CONTEXT_CHARS
from app.logger import get_logger

logger = get_logger("context_builder")


def _clean(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Assemble chunks into a formatted context string.
    - Cleans each chunk
    - Deduplicates by normalized text
    - Respects character budget
    - Labels each chunk with source and relevance score
    """
    seen_texts: set[str] = set()
    parts: list[str] = []
    total = 0

    for i, chunk in enumerate(chunks, 1):
        text = _clean(chunk.get("text", ""))
        normalized = text.lower()

        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        source = chunk.get("source", "unknown")
        score = chunk.get("rerank_score", chunk.get("score", "N/A"))
        block = f"[{i}] Source: {source} | Score: {score}\n{text}"

        if total + len(block) > max_chars:
            logger.info(f"Context budget reached at chunk {i}. Stopping.")
            break

        parts.append(block)
        total += len(block)

    context = "\n\n".join(parts)
    logger.info(f"Context built: {len(parts)} chunks, {len(context)} chars")
    return context
