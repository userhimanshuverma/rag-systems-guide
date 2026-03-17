# RAG Systems Guide — 21 Day Series

A structured 21-day learning series on Retrieval Augmented Generation (RAG), from core concepts to production-ready systems.

---

## What is RAG?

RAG (Retrieval Augmented Generation) is a technique that gives LLMs access to external knowledge at inference time — instead of relying solely on what they learned during training. The idea is simple: before generating an answer, retrieve relevant documents, then pass them as context to the LLM.

---

## Folder Structure

```
rag-systems-guide/
¦
+-- phase-1-foundations/                          # Why RAG exists
¦   +-- day-01-why-llms-need-external-knowledge/
¦   +-- day-02-what-is-rag/
¦   +-- day-03-rag-vs-fine-tuning/
¦   +-- day-04-basic-rag-pipeline/
¦   +-- day-05-where-basic-rag-fails/
¦
+-- phase-2-how-retrieval-works/                  # The retrieval engine
¦   +-- day-06-embeddings-explained/
¦   +-- day-07-vector-databases/
¦   +-- day-08-similarity-search/
¦   +-- day-09-document-chunking/
¦   +-- day-10-metadata-filtering/
¦   +-- day-11-hybrid-search/
¦   +-- day-12-reranking/
¦
+-- phase-3-building-rag-system/                  # Hands-on building
¦   +-- day-13-document-ingestion/
¦   +-- day-14-indexing-pipelines/
¦   +-- day-15-query-pipelines/
¦   +-- day-16-context-construction/
¦   +-- day-17-prompting-with-context/
¦
+-- phase-4-advanced-architectures/               # Advanced patterns
    +-- day-18-multi-query-retrieval/
    +-- day-19-agentic-rag/
    +-- day-20-graph-rag/
    +-- day-21-production-rag-systems/
```

---

## Series Breakdown

### Phase 1 — Foundations of RAG (Days 1–5)
> Why RAG exists and what problem it solves

| Day | Topic |
|-----|-------|
| 01 | Why LLMs Need External Knowledge — hallucinations, outdated knowledge, limits of static training |
| 02 | What RAG Actually Is — the simple idea: retrieve knowledge before generating answers |
| 03 | RAG vs Fine-Tuning — when to retrieve vs when to retrain |
| 04 | The Basic RAG Pipeline — User Query ? Retriever ? Documents ? LLM ? Answer |
| 05 | Where Basic RAG Fails — bad retrieval, irrelevant context, token limits, hallucinations |

---

### Phase 2 — How Retrieval Actually Works (Days 6–12)
> The engine under the hood

| Day | Topic |
|-----|-------|
| 06 | Embeddings Explained — turning text into numerical meaning vectors |
| 07 | Vector Databases — why traditional databases are not enough for semantic search |
| 08 | Similarity Search — finding relevant documents using vector distance |
| 09 | Document Chunking Strategies — why large documents must be split for effective retrieval |
| 10 | Metadata Filtering — narrowing search using tags, sources, timestamps, categories |
| 11 | Hybrid Search — combining keyword search with vector search for better recall |
| 12 | Reranking — using a second model to reorder retrieved documents by relevance |

---

### Phase 3 — Building a Real RAG System (Days 13–17)
> From raw files to working pipelines

| Day | Topic |
|-----|-------|
| 13 | Document Ingestion Pipelines — how raw files become structured knowledge |
| 14 | Indexing Pipelines — embedding documents and storing vectors efficiently |
| 15 | Query Pipelines — what happens from user question to retrieved documents |
| 16 | Context Construction — how retrieved chunks are assembled for the LLM |
| 17 | Prompting with Retrieved Knowledge — guiding the LLM to use context properly |

---

### Phase 4 — Advanced RAG Architectures (Days 18–21)
> Pushing RAG further

| Day | Topic |
|-----|-------|
| 18 | Multi-Query Retrieval — generating multiple search queries to improve recall |
| 19 | Agentic RAG — AI agents dynamically deciding what to retrieve and when |
| 20 | Graph RAG — using knowledge graphs to retrieve connected information |
| 21 | Production RAG Systems — scaling, caching, monitoring, evaluation, latency, and cost |

---

## How to Use This Repo

Each day folder contains the content, code, and notes for that topic. Work through them in order — each day builds on the previous one.

- Start with Phase 1 if you are new to RAG
- Jump to Phase 2 if you understand the concept but want to know how retrieval works
- Go to Phase 3 if you are ready to build
- Phase 4 is for when you want to go beyond the basics

---

## Prerequisites

- Basic understanding of how LLMs work (GPT, Claude, etc.)
- Python basics (most examples will be in Python)
- No prior RAG experience needed
