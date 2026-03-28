# Day 12 — Reranking

> "Getting the right documents is step one. Getting them in the right order is step two."

---

## The Problem

Your retriever is working. It pulls back 20 document chunks that are all somewhat related to the user's query.

But "somewhat related" isn't good enough.

The LLM has a limited context window. You can only pass in the top 3, 5, maybe 10 chunks. If the most relevant chunk is sitting at position 15 in your retrieval results — buried under 14 less relevant ones — it never makes it into the prompt. The LLM answers without the most important piece of information.

Retrieval gets you a broad set of candidates. Reranking makes sure the best ones rise to the top.

---

## What Reranking Actually Is

Retrieval is fast but approximate. It uses vector similarity to quickly find documents that are in the right neighborhood — but it doesn't deeply analyze how well each document actually answers the specific question.

Reranking is a second pass. After retrieval gives you a pool of candidates, a reranker model reads each document alongside the query and scores them more carefully — based on actual relevance, not just vector proximity.

The result is a reordered list where the most genuinely useful documents are at the top.

> Retrieval = cast a wide net, pull in candidates fast
> Reranking = carefully sort those candidates by true relevance

---

## The Google Analogy

When you search on Google, two things happen:

First, Google retrieves thousands of potentially relevant pages from its index — fast, at scale.

Then, a ranking algorithm scores and orders those pages. The ones that best match your intent, have the most authority, and are most likely to answer your question end up at the top.

You only see the top 10. But Google considered thousands.

Reranking in RAG works the same way. The retriever is the fast first pass. The reranker is the careful second pass that decides what actually gets shown to the LLM.

---

## How It Works Step by Step

**Step 1 — Retrieval runs first**

The retriever pulls the top K candidates from the vector database. K is intentionally set higher than what you'll ultimately pass to the LLM — you want a broad pool to work with.

```
Query: "How do I cancel my enterprise subscription?"
Retriever returns top 20 chunks (K=20)
```

**Step 2 — Reranker scores each candidate**

The reranker model takes each retrieved chunk and the original query, reads them together, and assigns a relevance score. This is a deeper analysis than vector similarity — it considers the actual content of the document in relation to the specific question.

```
Chunk 1: "Enterprise plans can be cancelled via the billing portal..." → score: 0.94
Chunk 2: "Subscription management overview..." → score: 0.71
Chunk 3: "How to upgrade your plan..." → score: 0.43
Chunk 4: "Cancellation policy for all plans..." → score: 0.89
...
```

**Step 3 — Documents are reordered by score**

The chunks are sorted from highest to lowest relevance score.

```
Reranked order:
1. "Enterprise plans can be cancelled via the billing portal..." (0.94)
2. "Cancellation policy for all plans..." (0.89)
3. "Subscription management overview..." (0.71)
...
```

**Step 4 — Top N passed to LLM**

Only the top N chunks after reranking are passed to the LLM as context. N is smaller than K — you retrieved broadly, now you're passing only the best.

```
Top 5 reranked chunks → LLM → Final Answer
```

---

## Where Reranking Fits in the RAG Pipeline

```
User Query
    ↓
Retriever
(fast, broad — returns top K candidates)
    ↓
Top K Chunks (e.g. 20 results)
    ↓
Reranker Model
(slow, precise — scores each chunk against query)
    ↓
Reordered + Filtered Results (top N, e.g. 5)
    ↓
LLM → Final Answer
```

Reranking sits between the retriever and the LLM. It's a quality filter — taking the retriever's broad output and refining it into a tight, high-quality context window.

---

## Full Flow Diagram

```
User Query
    ↓
Vector DB / Hybrid Search
    ↓
Top K Retrieved Chunks (broad pool)
    ↓
┌─────────────────────────────────────┐
│           Reranker Model            │
│                                     │
│  Query + Chunk 1  → score: 0.94     │
│  Query + Chunk 2  → score: 0.71     │
│  Query + Chunk 3  → score: 0.43     │
│  Query + Chunk 4  → score: 0.89     │
│  ...                                │
└─────────────────────────────────────┘
    ↓
Sorted by Score (highest first)
    ↓
Top N Chunks (precision-filtered)
    ↓
LLM → Final Answer
```

---

## Retrieval vs Reranking — The Key Distinction

This is worth being very clear about because people often confuse the two.

| | Retrieval | Reranking |
|---|---|---|
| Goal | Find candidates | Sort candidates by relevance |
| Speed | Fast (milliseconds) | Slower (reads each doc carefully) |
| Method | Vector similarity / keyword match | Deep relevance scoring |
| Optimizes for | Recall (don't miss relevant docs) | Precision (surface the best ones) |
| Input | Query | Query + each retrieved chunk |
| Output | Unordered pool of candidates | Ordered list, best first |

**Recall** means finding everything that could be relevant — you don't want to miss good documents.
**Precision** means making sure what you pass to the LLM is actually the best material available.

Retrieval handles recall. Reranking handles precision. You need both.

---

## Why It Matters

**The LLM only sees what you give it**
If the most relevant chunk is ranked 12th after retrieval and you only pass the top 5 to the LLM — that chunk is invisible. Reranking fixes this by ensuring the best material rises to the top before the LLM ever sees it.

**Vector similarity isn't the same as relevance**
Two chunks can be semantically close to a query without actually answering it. A reranker reads the content more carefully and distinguishes between "related to the topic" and "actually answers the question."

**It compensates for retrieval imperfections**
No retrieval system is perfect. Reranking adds a safety net — even if retrieval returns some mediocre results, the reranker pushes the good ones up and the weak ones down.

**Better context = better answers**
The LLM's output quality is directly tied to the quality of its input context. Reranking is one of the highest-leverage improvements you can make to a RAG system because it directly improves what the LLM works with.

---

## The Trade-off

Reranking is more computationally expensive than retrieval. The reranker model reads every candidate chunk paired with the query — that's more work than a simple vector comparison.

This is why the two-stage approach exists:
- Stage 1 (retrieval): fast, broad, cheap — get 20-50 candidates
- Stage 2 (reranking): slower, precise, more expensive — score and sort those candidates

You don't rerank your entire knowledge base. You rerank the small pool that retrieval already narrowed down. That keeps the cost manageable while still getting the precision benefit.

---

## Key Takeaways

- Retrieval is fast but approximate — it finds candidates, not necessarily the best ones
- Reranking is a second pass that scores each retrieved chunk against the query for true relevance
- Retrieval optimizes for recall (don't miss relevant docs); reranking optimizes for precision (surface the best ones)
- The reranker reads query + document together — a deeper analysis than vector similarity alone
- Only the top N reranked chunks are passed to the LLM, keeping context tight and high quality
- Reranking is more expensive than retrieval, so it runs on a small candidate pool — not the full knowledge base
- Better context going into the LLM means better answers coming out
- Reranking is one of the highest-impact improvements you can add to a basic RAG system

---

## What's Next

Phase 2 is complete. You now understand the full retrieval stack — embeddings, vector databases, similarity search, metadata filtering, hybrid search, and reranking.

Phase 3 starts on **Day 13**, and it's time to build.

We'll start with **Document Ingestion Pipelines** — how raw files (PDFs, Word docs, web pages, databases) get transformed into structured, searchable knowledge that your RAG system can actually use.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
