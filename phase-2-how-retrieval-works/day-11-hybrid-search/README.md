# Day 11 — Hybrid Search

> "Neither keyword search nor semantic search is perfect alone. Together, they cover each other's blind spots."

---

## The Problem

By now you know that semantic search is powerful. It understands meaning, handles synonyms, and finds relevant content even when the exact words don't match.

But it has a blind spot.

Ask it to find *"error code 404"* or *"function getUserById"* or *"invoice #INV-2024-0892"* — and it might struggle. These are exact terms. Specific identifiers. Technical strings where the precise wording matters more than the meaning.

Semantic search is built for meaning. It's not built for exact matches.

On the flip side, keyword search is great at exact matches. But ask it to find documents about *"ways to reduce server costs"* when your documents talk about *"cloud infrastructure optimization"* — and it finds nothing. The words don't overlap.

Both approaches have real strengths. Both have real weaknesses. And in production systems, you'll encounter queries that need both.

That's what hybrid search solves.

---

## The Simple Explanation

Hybrid search combines two retrieval methods and merges their results:

> **Semantic search** — finds documents by meaning (vector similarity)
> **Keyword search** — finds documents by exact word matches (text matching)

Run both in parallel. Combine the results. Rank them together. Return the best of both worlds.

The result is a retrieval system that's more robust than either approach alone — one that handles both conceptual questions and precise lookups without having to choose between them.

---

## The Google Analogy

Think about how Google actually works when you search.

If you type *"best way to handle async errors in JavaScript"* — Google understands the intent. It finds pages about error handling, promises, async/await patterns — even if they don't use your exact phrase. That's semantic understanding at work.

But if you type *"TypeError: Cannot read properties of undefined"* — Google finds pages that contain that exact error message. Exact match matters here. You want the specific string, not a conceptual interpretation.

Google doesn't choose one mode or the other. It blends both — understanding your intent while also respecting exact terms when they matter.

Hybrid search in RAG works the same way. It doesn't force you to pick semantic or keyword. It runs both and combines the results intelligently.

---

## How It Works Step by Step

**Step 1 — Query comes in**

```
User query: "What does error code 503 mean in our API?"
```

**Step 2 — Two searches run in parallel**

*Semantic search:*
The query is embedded into a vector. The vector database finds the most semantically similar chunks — documents about API errors, service availability, HTTP status codes.

*Keyword search:*
The query is broken into terms. A text index finds chunks that contain the exact words — "503", "error code", "API".

```
Semantic results:          Keyword results:
1. API error handling      1. "503 Service Unavailable"
2. HTTP status codes       2. "error code 503 returned when..."
3. Service availability    3. API troubleshooting guide
4. Retry logic docs        4. Status code reference
```

**Step 3 — Results are merged**

The two result sets are combined. Documents that appear in both lists get a boost — they're relevant by both meaning and exact match, which is a strong signal.

**Step 4 — Unified ranking**

A scoring function combines the semantic similarity score and the keyword relevance score into a single unified rank. The top K results from this combined ranking are selected.

**Step 5 — Final results passed to LLM**

The merged, ranked chunks become the retrieved context for the LLM.

---

## Where Hybrid Search Fits in the RAG Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────┐
│           Hybrid Search Layer           │
│                                         │
│  Semantic Search    Keyword Search      │
│  (vector similarity) (text matching)    │
│         ↓                  ↓            │
│         └──────┬───────────┘            │
│                ↓                        │
│         Merge + Re-score                │
└─────────────────────────────────────────┘
    ↓
Combined Ranked Results
    ↓
Top K Chunks
    ↓
LLM → Final Answer
```

Hybrid search sits inside the retriever layer. From the outside, the pipeline looks the same — query in, relevant chunks out. The hybrid logic is internal to the retrieval step.

---

## Full Flow Diagram

```
User Query
    ↓
    ├──────────────────────┬──────────────────────┐
    ↓                      ↓                      │
Embedding Model        Text Index                 │
    ↓                      ↓                      │
Query Vector           Keyword Terms              │
    ↓                      ↓                      │
Vector DB Search       Keyword Search             │
(semantic results)     (exact match results)      │
    ↓                      ↓                      │
    └──────────────────────┘                      │
                ↓                                 │
         Merge Results                            │
                ↓                                 │
         Unified Ranking                          │
                ↓                                 │
         Top K Chunks → LLM → Answer ─────────────┘
```

---

## Why Each Approach Alone Falls Short

| Situation | Semantic Search | Keyword Search |
|---|---|---|
| "Ways to reduce cloud costs" | Finds relevant docs | Misses if exact words differ |
| "Error code 404" | May miss exact code | Finds it directly |
| "getUserById function" | May not understand code | Finds exact function name |
| "Invoice #INV-2024-0892" | Likely misses | Finds exact match |
| "How do I handle timeouts?" | Understands intent | May miss synonyms |

Hybrid search handles all of these. Neither approach alone does.

---

## Why It Matters

**Better recall**
Recall measures how many relevant documents you actually find. Hybrid search finds more relevant documents than either approach alone — semantic catches the conceptual matches, keyword catches the exact ones.

**Better precision**
Precision measures how many of your returned results are actually relevant. By combining scores, hybrid search filters out results that are only weakly relevant to either method.

**More robust across query types**
Users ask all kinds of questions — vague conceptual ones, precise technical ones, and everything in between. Hybrid search handles the full spectrum without needing to know in advance what type of query is coming.

**Essential for technical and enterprise content**
Code, error messages, product IDs, legal clause numbers, API endpoints — these are all exact-match scenarios that semantic search handles poorly. Hybrid search makes RAG viable for technical and enterprise use cases.

---

## Real-World Examples

**Code search**
A developer asks: *"Where is the `processPayment` function defined?"*
Keyword search finds the exact function name. Semantic search finds related payment processing logic. Together they return the most relevant code chunks.

**Log analysis**
An engineer asks: *"Show me logs with OutOfMemoryError from the auth service."*
Keyword search finds exact error strings. Semantic search finds related memory and service failure logs. Hybrid gives the full picture.

**Enterprise knowledge base**
An employee asks: *"What's the policy on remote work for contractors?"*
Semantic search finds policy documents about flexible work arrangements. Keyword search finds documents that explicitly mention "contractors." Hybrid returns the most relevant policy chunks.

**Customer support**
A user asks: *"I'm getting error 422 on the checkout page."*
Keyword search finds the exact error code in documentation. Semantic search finds related checkout and validation error content. The LLM gets both.

---

## Key Takeaways

- Semantic search understands meaning but can miss exact terms and identifiers
- Keyword search finds exact matches but misses conceptual and synonym-based queries
- Hybrid search runs both in parallel and merges the results into a unified ranking
- Documents that appear in both result sets get a relevance boost
- Hybrid search improves both recall (finding more relevant docs) and precision (filtering irrelevant ones)
- It's especially important for technical content — code, error codes, IDs, API names
- From the pipeline's perspective, hybrid search is still just a retriever — it takes a query and returns ranked chunks
- Most production RAG systems benefit from hybrid search over pure semantic search alone

---

## What's Next

Hybrid search gives you a better pool of retrieved documents. But the ranking that comes out of merging two result sets isn't always perfect.

On **Day 12**, we'll cover Reranking — using a second, more powerful model to re-score and reorder retrieved documents by true relevance, and why this final step can dramatically improve the quality of what the LLM receives.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
