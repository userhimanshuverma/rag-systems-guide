"""
Document ingestion + FAISS indexing script.

Usage:
    python scripts/ingest.py --data_dir data/docs

Steps:
1. Load .txt files from data_dir
2. Clean and chunk each document
3. Embed chunks using sentence-transformers
4. Build a FAISS index (Inner Product = cosine similarity with normalized vectors)
5. Save index + metadata to disk
"""

import os
import re
import json
import argparse
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP
)
from app.logger import get_logger

logger = get_logger("ingest")


def load_documents(folder: str) -> list[dict]:
    """Load all .txt files from a folder."""
    docs = []
    for path in Path(folder).glob("*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"text": f.read(), "source": path.name})
    logger.info(f"Loaded {len(docs)} documents from {folder}")
    return docs


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index(docs: list[dict], model: SentenceTransformer) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Embed all chunks and build a FAISS Inner Product index."""
    all_chunks = []
    chunk_id = 0

    for doc in docs:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "source": doc["source"],
                "chunk_index": i
            })
            chunk_id += 1

    logger.info(f"Total chunks to embed: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    logger.info("Embedding chunks (this may take a moment)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (with normalized vectors)
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, all_chunks


def save(index: faiss.IndexFlatIP, metadata: list[dict]):
    """Persist FAISS index and metadata to disk."""
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Index saved to {FAISS_INDEX_PATH}")
    logger.info(f"Metadata saved to {METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS index")
    parser.add_argument("--data_dir", default="data/docs", help="Folder containing .txt files")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", args.data_dir)

    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Create data/docs/ and add .txt files, then re-run.")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    docs = load_documents(data_dir)
    if not docs:
        logger.error("No .txt files found. Add documents to data/docs/ and re-run.")
        sys.exit(1)

    index, metadata = build_index(docs, model)
    save(index, metadata)

    logger.info(f"Ingestion complete. {len(metadata)} chunks indexed from {len(docs)} documents.")


if __name__ == "__main__":
    main()
