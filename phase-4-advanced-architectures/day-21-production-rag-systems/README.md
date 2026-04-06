# Day 21 — Production RAG Systems

> "A prototype proves the concept. A production system proves the engineering."

This is the final day of the 21-Day RAG Series. Everything you've learned — embeddings, vector search, chunking, reranking, context construction, prompting, multi-query, agentic RAG, graph RAG — comes together here in a complete, runnable production system.

---

## What This Is

A production-grade RAG system with:

- FastAPI REST API (sync + async endpoints)
- FAISS vector index for fast semantic search
- Hybrid retrieval (vector + optional keyword filter)
- Multi-query retrieval for better recall
- Weighted reranking (vector score + keyword overlap)
- Context construction with deduplication and token budget
- Strict grounding prompts to minimize hallucination
- Local Mistral LLM inference via Ollama (no paid APIs)
- SQLite-backed query cache
- Structured logging with rotating file handler
- Evaluation pipeline with keyword recall and source hit metrics
- Docker support

---

## Project Structure

```
day-21-production-rag-systems/
│
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + pipeline orchestration
│   ├── config.py         # All configuration in one place
│   ├── retriever.py      # FAISS-based semantic retriever
│   ├── reranker.py       # Weighted reranking (vector + keyword)
│   ├── context_builder.py # Context assembly with token budget
│   ├── prompt.py         # Prompt templates
│   ├── llm.py            # Ollama/Mistral inference
│   ├── cache.py          # SQLite query cache
│   ├── evaluator.py      # Evaluation pipeline
│   └── logger.py         # Centralised logging
│
├── scripts/
│   └── ingest.py         # Document ingestion + FAISS indexing
│
├── data/
│   ├── docs/             # Put your .txt documents here
│   └── eval_dataset.json # Test cases for evaluation
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Architecture

```
                    INDEXING (run once)
                    ──────────────────
.txt files in data/docs/
        ↓
  scripts/ingest.py
  Load → Clean → Chunk → Embed → FAISS Index
        ↓
  data/faiss.index + data/metadata.json


                    QUERY (per request)
                    ───────────────────
POST /query
        ↓
  Retriever (FAISS semantic search)
  + optional keyword filter
  + optional multi-query variations
        ↓
  Reranker (vector score × 0.7 + keyword overlap × 0.3)
        ↓
  Context Builder (deduplicate + token budget)
        ↓
  Cache check (SQLite, keyed by query+context hash)
        ↓
  Prompt Builder (strict grounding instructions)
        ↓
  LLM (local Mistral via Ollama)
        ↓
  Cache store + Response
```

---

## Setup

### 1. Install Python dependencies

```bash
cd day-21-production-rag-systems
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download Ollama from https://ollama.com and install it.

```bash
# Pull the Mistral model (~4GB)
ollama pull mistral

# Start the Ollama server (keep this running)
ollama serve
```

### 3. Add your documents

Place `.txt` files in `data/docs/`. Two sample documents are already included.

### 4. Run ingestion

```bash
python scripts/ingest.py --data_dir data/docs
```

This loads your documents, chunks them, embeds them, and saves the FAISS index to `data/`.

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now running at `http://localhost:8000`.

---

## API Endpoints

### POST /query

Main RAG query endpoint.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG and how does it reduce hallucinations?",
    "top_k": 10,
    "top_n": 4,
    "use_multi_query": false,
    "use_cache": true
  }'
```

Response:
```json
{
  "query": "What is RAG and how does it reduce hallucinations?",
  "answer": "RAG (Retrieval Augmented Generation) reduces hallucinations by...",
  "chunks_used": 4,
  "sources": ["rag_overview.txt"],
  "latency_s": 2.341,
  "cache_hit": false
}
```

### POST /query/batch

Process multiple queries concurrently.

```bash
curl -X POST http://localhost:8000/query/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is chunking?", "How does reranking work?"],
    "top_k": 10,
    "top_n": 4
  }'
```

### GET /health

System health check — LLM availability, index status, cache stats.

```bash
curl http://localhost:8000/health
```

### DELETE /cache

Clear the query cache.

```bash
curl -X DELETE http://localhost:8000/cache
```

### POST /evaluate

Run the evaluation pipeline against `data/eval_dataset.json`.

```bash
curl -X POST http://localhost:8000/evaluate
```

Results are logged to `data/rag.log`.

---

## Example Queries

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'

# With multi-query for better recall
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does retrieval work?", "use_multi_query": true}'

# With keyword filter (hybrid-style)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "advanced retrieval techniques", "keyword_filter": "hybrid"}'
```

---

## Module Explanations

| Module | Responsibility |
|---|---|
| `config.py` | All tuneable parameters — chunk size, top_k, model names, paths |
| `retriever.py` | Loads FAISS index, embeds queries, returns top-K chunks |
| `reranker.py` | Re-scores chunks using vector score + keyword overlap |
| `context_builder.py` | Assembles chunks into clean context with dedup + token budget |
| `prompt.py` | Prompt templates with strict grounding instructions |
| `llm.py` | Calls local Mistral via Ollama REST API |
| `cache.py` | SQLite cache keyed by SHA-256 hash of query + context |
| `evaluator.py` | Runs test cases, measures recall and source hit rate |
| `logger.py` | Rotating file + console logging for all modules |
| `main.py` | FastAPI app, pipeline orchestration, all endpoints |
| `scripts/ingest.py` | Load → clean → chunk → embed → FAISS index → save |

---

## Configuration

All settings are in `app/config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K_RETRIEVAL` | `10` | Retrieval pool size |
| `TOP_N_RERANK` | `4` | Chunks passed to LLM |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `60` | Overlap between chunks |
| `LLM_MODEL` | `mistral` | Ollama model name |
| `MAX_CONTEXT_CHARS` | `2000` | Context window budget |

Override via environment variables or `.env` file:

```env
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=mistral
```

---

## Docker

```bash
# Build
docker build -t production-rag .

# Run (Ollama must be accessible from the container)
docker run -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  -v $(pwd)/data:/app/data \
  production-rag
```

---

## Evaluation

Add test cases to `data/eval_dataset.json`:

```json
[
  {
    "query": "What is RAG?",
    "expected_answer": "RAG stands for Retrieval Augmented Generation...",
    "expected_source": "rag_overview.txt"
  }
]
```

Then run:

```bash
curl -X POST http://localhost:8000/evaluate
```

Metrics logged:
- `avg_answer_recall` — keyword overlap between expected and actual answer
- `source_hit_rate` — fraction of queries where the expected source was retrieved
- `avg_latency_s` — average end-to-end latency

---

## Production Considerations

**Scaling**
- Replace FAISS flat index with `IndexIVFFlat` for million-scale vectors
- Use a proper vector database (Qdrant, Weaviate, Pinecone) for persistence and filtering
- Deploy behind a load balancer with multiple API instances

**Caching**
- Current SQLite cache works for single-instance deployments
- For multi-instance: replace with Redis

**Monitoring**
- Current: structured logs to rotating file
- Production: ship logs to Datadog, CloudWatch, or ELK stack
- Add Prometheus metrics endpoint for latency histograms and error rates

**LLM**
- Current: local Mistral via Ollama (great for development and privacy)
- Production options: vLLM for GPU-accelerated inference, or a managed API

**Security**
- Add API key authentication to all endpoints
- Sanitize user inputs before embedding and prompting
- Rate limit the `/query` endpoint

---

## What You've Built Over 21 Days

| Phase | Days | What You Learned |
|---|---|---|
| Foundations | 1–5 | Why RAG exists, what it is, when to use it, how it fails |
| Retrieval | 6–12 | Embeddings, vector DBs, similarity search, chunking, filtering, hybrid, reranking |
| Building | 13–17 | Ingestion, indexing, query pipeline, context construction, prompting |
| Advanced | 18–21 | Multi-query, agentic RAG, graph RAG, production systems |

You now understand RAG from first principles to production deployment.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
