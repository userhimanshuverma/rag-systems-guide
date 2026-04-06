"""
Retriever module.
Loads a FAISS index + metadata, embeds queries, and returns top-K results.
Supports optional keyword pre-filtering for hybrid-style retrieval.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH,
    TOP_K_RETRIEVAL, RELEVANCE_THRESHOLD
)
from app.logger import get_logger

logger = get_logger("retriever")


class Retriever:
    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._load_index()

    def _load_index(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            logger.warning("FAISS index not found. Run scripts/ingest.py first.")
            return
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.metadata)} chunks")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32).reshape(1, -1)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL, keyword_filter: str = "") -> list[dict]:
        """
        Retrieve top-K chunks for a query.
        Optional keyword_filter: only return chunks whose text contains this string.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.error("Index is empty or not loaded.")
            return []

        query_vec = self._embed(query)
        scores, indices = self.index.search(query_vec, min(top_k * 2, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < RELEVANCE_THRESHOLD:
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = round(float(score), 4)

            # Optional keyword filter (hybrid-style)
            if keyword_filter and keyword_filter.lower() not in chunk["text"].lower():
                continue

            results.append(chunk)
            if len(results) >= top_k:
                break

        logger.info(f"Retrieved {len(results)} chunks for query='{query[:60]}'")
        return results

    def multi_query_retrieve(self, queries: list[str], top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Run retrieval for multiple query variations and merge results.
        Deduplicates by chunk_id, keeping the highest score.
        """
        best: dict[int, dict] = {}
        for q in queries:
            for chunk in self.retrieve(q, top_k=top_k):
                cid = chunk["chunk_id"]
                if cid not in best or chunk["score"] > best[cid]["score"]:
                    best[cid] = chunk

        merged = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        logger.info(f"Multi-query: {len(queries)} queries -> {len(merged)} unique chunks")
        return merged
