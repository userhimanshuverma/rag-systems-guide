"""
Central configuration for the production RAG system.
All tuneable parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Retrieval ---
TOP_K_RETRIEVAL = 10       # broad retrieval pool
TOP_N_RERANK = 4           # final chunks passed to LLM
RELEVANCE_THRESHOLD = 0.30 # minimum cosine similarity to include a result

# --- Chunking ---
CHUNK_SIZE = 400           # characters per chunk
CHUNK_OVERLAP = 60         # overlap between consecutive chunks

# --- FAISS index ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

# --- Cache ---
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache.db")

# --- Logging ---
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rag.log")

# --- LLM (Ollama local) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
LLM_TIMEOUT = 120          # seconds

# --- Context ---
MAX_CONTEXT_CHARS = 2000
