# Day 10 — Metadata Filtering

> "Searching everything when you only need a slice is just noise with extra steps."

---

## The Problem

Imagine your RAG system has ingested 50,000 document chunks. Product manuals, support tickets, legal contracts, HR policies, release notes — all mixed together in one vector database.

A user asks: *"What changed in the API in the last release?"*

Without any filtering, the similarity search scans all 50,000 chunks. It might return a mix of old release notes, unrelated product docs, and HR policies that happen to use similar words. The LLM gets noisy, irrelevant context — and the answer suffers.

The problem isn't the search. The problem is the search space is too large and too mixed.

Metadata filtering solves this by narrowing the search before similarity search even runs.

---

## What Metadata Actually Is

Metadata is additional information *about* a document — not the content itself, but the context around it.

Think of it like labels attached to each document chunk when you store it:

```
Chunk: "The /users endpoint now returns a pagination token..."
Metadata:
  - source: "api-release-notes"
  - date: "2024-11-15"
  - category: "engineering"
  - version: "v3.2"
  - department: "product"
```

The chunk is the content. The metadata is everything that describes where it came from, when it was created, what type it is, and who it belongs to.

When stored in a vector database, both the vector and the metadata travel together. This means you can filter by metadata before — or alongside — similarity search.

---

## Common Types of Metadata

**Source / Origin**
Which document, file, or system did this chunk come from?
`source: "employee-handbook"`, `source: "support-tickets"`, `source: "legal-contracts"`

**Timestamp / Date**
When was this document created or last updated?
`date: "2024-10-01"`, `last_updated: "2025-01-15"`

**Category / Type**
What kind of document is this?
`category: "policy"`, `type: "release-notes"`, `type: "faq"`

**Tags**
Free-form labels for flexible filtering.
`tags: ["billing", "enterprise", "v2"]`

**Department / Team**
Who owns this content?
`department: "engineering"`, `team: "customer-success"`

**Language**
What language is the content in?
`language: "en"`, `language: "fr"`

Any structured attribute you can attach to a document becomes a potential filter.

---

## How It Works Step by Step

**Step 1 — User query arrives with context**

Sometimes the filter comes from the query itself. Sometimes it comes from the user's session context (e.g., they're logged in as an enterprise customer, or they've selected a specific product).

```
User query: "What changed in the API in the last release?"
Inferred filter: category = "release-notes", date = recent
```

**Step 2 — Metadata filter is applied**

Before similarity search runs, the vector database applies the filter. It removes all chunks that don't match the criteria.

```
All 50,000 chunks
        ↓
Filter: category = "release-notes" AND date >= "2024-10-01"
        ↓
Reduced to 340 chunks
```

**Step 3 — Similarity search runs on the reduced set**

Now similarity search only compares the query vector against the 340 filtered chunks — not all 50,000. It's faster, more focused, and far less noisy.

**Step 4 — Top results returned**

The top K most similar chunks from the filtered set are returned as context for the LLM.

---

## The Google Search Analogy

You've used Google's search filters without thinking about it.

When you search for something and then click "Past year" — that's metadata filtering. You're telling the search engine: *"Only show me results from the last 12 months."*

When you add `site:docs.python.org` to a search — that's metadata filtering by source.

When you filter image results by color, size, or license type — metadata filtering again.

Google doesn't re-crawl the internet for each filter. It applies the filter to its existing index, narrows the candidate set, then ranks within that set.

RAG metadata filtering works exactly the same way. Filter first, search within the filtered set, return the best results.

---

## Where Filtering Fits in the RAG Pipeline

```
User Query + Context
        ↓
Extract or infer metadata filters
(from query, user session, or explicit selection)
        ↓
Apply Metadata Filter to Vector DB
(remove non-matching chunks)
        ↓
Reduced Candidate Set
        ↓
Similarity Search
(semantic search within filtered set)
        ↓
Top K Relevant Chunks
        ↓
LLM → Final Answer
```

Filtering happens before similarity search. It shrinks the search space so the semantic search is both faster and more accurate.

---

## Full Flow Diagram

```
User Query
    ↓
┌─────────────────────────────────┐
│  Metadata Filter                │
│  e.g. date > 2024-10-01         │
│       category = release-notes  │
└─────────────────────────────────┘
    ↓
Reduced Dataset (340 of 50,000 chunks)
    ↓
Similarity Search
    ↓
Top K Results
    ↓
LLM → Answer
```

---

## Why It Matters

**Better accuracy**
Filtering removes irrelevant documents before search even starts. The LLM receives context that's not just semantically similar — it's also contextually appropriate (right source, right time, right type).

**Faster retrieval**
Similarity search over 300 chunks is orders of magnitude faster than over 50,000. At scale, this difference is significant.

**Less noise in context**
Without filtering, a broad similarity search might return chunks from completely different domains that happen to share vocabulary. Filtering keeps the context clean and focused.

**Enables multi-tenant systems**
In systems where different users should only see their own data, metadata filtering enforces data isolation. User A's query only searches User A's documents.

**Supports time-sensitive queries**
For questions about recent events, policies, or releases, date filtering ensures the LLM works with current information — not outdated content that happens to be semantically similar.

---

## Real-World Example

**Scenario: Engineering incident response system**

Your company ingests logs, runbooks, post-mortems, and monitoring alerts into a RAG system. An on-call engineer asks:

*"What caused the database slowdown last Tuesday?"*

Without metadata filtering:
- Similarity search returns chunks from old post-mortems, general database docs, and unrelated alerts
- The LLM gets confused context spanning months of history
- The answer is vague and unhelpful

With metadata filtering:
- Filter: `system = "database"`, `date >= last Tuesday`, `type = "alert OR log"`
- Similarity search runs on a small, focused set of recent database-related chunks
- The LLM gets precise, relevant context
- The answer is specific and actionable

Same query. Same LLM. Completely different outcome — just from filtering the search space.

---

## Key Takeaways

- Metadata is structured information attached to document chunks — source, date, category, tags, etc.
- Metadata filtering narrows the search space before similarity search runs
- Filtering improves accuracy, speed, and signal-to-noise ratio in retrieval
- Filters can be applied explicitly (user selects a filter) or inferred (from query context or session)
- Date filtering keeps answers current; source filtering keeps answers scoped; category filtering keeps answers relevant
- In multi-tenant systems, metadata filtering enforces data isolation between users
- Filtering and similarity search work together — they're not alternatives, they're complements

---

## What's Next

Metadata filtering narrows the search space. But what if you need both semantic understanding *and* exact keyword matching in the same search?

On **Day 11**, we'll cover Hybrid Search — combining vector-based semantic search with traditional keyword search to get the best of both worlds, and why this combination often outperforms either approach alone.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
