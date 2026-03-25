# Day 9 — Document Chunking Strategies: Splitting Documents the Smart Way

> "You don't read a whole book to answer one question. You flip to the right page."

---

## The Problem

Imagine you have a 50-page technical manual. A user asks:

> "What is the recommended operating temperature?"

If you feed the entire 50-page document into your retrieval system, here's what happens:

- The embedding for the whole document becomes a blurry average of everything in it
- The specific sentence about temperature gets buried under hundreds of unrelated sentences
- Retrieval returns the whole document — but the answer is one line on page 34
- The LLM gets overwhelmed with noise and may miss the actual answer

This is the core problem chunking solves.

---

## Simple Explanation

LLMs work best when given focused, relevant context — not walls of text.

When you embed a large document as a single unit, you lose precision. The vector that represents it tries to capture everything at once, which means it captures nothing well. It's like trying to describe an entire city with one word.

Smaller, focused pieces of text produce sharper, more meaningful embeddings. And sharper embeddings lead to better retrieval.

---

## What is Chunking?

Chunking is the process of splitting a large document into smaller, meaningful pieces before storing them in your vector database.

Each chunk:
- Represents a focused idea or section
- Gets its own embedding
- Can be retrieved independently
- Carries just enough context to be useful on its own

Instead of storing one giant blob, you store dozens (or hundreds) of precise, searchable units.

---

## The Analogy

Think about how you use a textbook.

When studying, you don't re-read the entire book every time you need to recall something. You highlight key paragraphs. You write notes in the margins. You bookmark specific sections.

When you need an answer, you go straight to the relevant highlighted paragraph — not the whole chapter.

Chunking does exactly this for your RAG system. It pre-highlights the document into focused, retrievable pieces so the system can go straight to what matters.

---

## How It Works (Step by Step)

**Step 1 — You have a raw document**

A PDF, a webpage, a support article, a legal contract. It's long, unstructured, and full of mixed content.

**Step 2 — Split it into chunks**

The document gets divided into smaller pieces. Each chunk is a few sentences or a paragraph — enough to carry a complete thought.

**Step 3 — Embed each chunk**

Each chunk is passed through an embedding model. The result is a vector that captures the meaning of that specific piece of text.

**Step 4 — Store in vector database**

Each chunk's vector (along with the original text) gets stored in the vector database, indexed and ready to search.

**Step 5 — Retrieval**

When a user asks a question, the query is embedded and compared against all chunk vectors. The closest matches are returned — not the whole document, just the relevant pieces.

**Step 6 — Generation**

Those retrieved chunks are passed to the LLM as context. The LLM now has focused, relevant information to generate a precise answer.

---

## Types of Chunking

There's no single right way to chunk. Here are the main approaches:

**Fixed-Size Chunking**

Split the document every N characters or N tokens, regardless of content boundaries.

- Simple and fast
- Doesn't respect sentence or paragraph structure
- Can cut a sentence in half, losing meaning at the edges
- Good starting point when you need something quick

```
[...chunk 1: 500 tokens...][...chunk 2: 500 tokens...][...chunk 3: 500 tokens...]
```

**Semantic Chunking**

Split based on meaning — at sentence boundaries, paragraph breaks, or section headers.

- Respects the natural structure of the document
- Each chunk contains a complete thought
- Produces better embeddings because the text is coherent
- Slightly more complex to implement

```
[Introduction paragraph] [Key concept paragraph] [Example paragraph] [Conclusion paragraph]
```

**Overlapping Chunks**

Add a small overlap between consecutive chunks — the last few sentences of chunk 1 also appear at the start of chunk 2.

- Prevents important context from being cut off at boundaries
- Helps when an answer spans two adjacent sections
- Increases storage slightly, but improves retrieval continuity

```
[chunk 1: sentences 1–10]
[chunk 2: sentences 8–18]   ← sentences 8–10 overlap
[chunk 3: sentences 16–26]  ← sentences 16–18 overlap
```

---

## Architecture Thinking: Where Chunking Fits

Chunking happens during the ingestion pipeline — before anything gets stored. It's a preprocessing step that shapes the quality of everything downstream.

```
Raw Document
     ↓
 [Chunking Layer]
  Split into N chunks
     ↓
 [Embedding Model]
  Each chunk → vector
     ↓
 [Vector Database]
  Store (chunk text + vector)
     ↓
 Ready for Retrieval
```

Get chunking wrong and your embeddings are noisy. Get it right and retrieval becomes precise and reliable.

This is why chunking is one of the highest-leverage decisions in a RAG system. It's not glamorous, but it quietly determines how well everything else works.

---

## ASCII Diagram

```
+---------------------------+
|      Raw Document         |
|  (50 pages, mixed content)|
+---------------------------+
             ↓
     [ Chunking Strategy ]
             ↓
+--------+ +--------+ +--------+
| Chunk1 | | Chunk2 | | Chunk3 |  ...more chunks
+--------+ +--------+ +--------+
             ↓
     [ Embedding Model ]
             ↓
+--------+ +--------+ +--------+
| Vec[1] | | Vec[2] | | Vec[3] |
+--------+ +--------+ +--------+
             ↓
     [ Vector Database ]
             ↓
+---------------------------+
|   Indexed & Searchable    |
+---------------------------+
             ↓
      User Query Arrives
             ↓
   Top-K Relevant Chunks Retrieved
             ↓
      LLM Generates Answer
```

---

## Tradeoffs

Chunking is a balancing act. Both extremes hurt you.

**Chunks too large:**
- Embeddings become vague and unfocused
- Retrieval returns too much noise
- LLM gets overwhelmed with irrelevant context
- Answers become less precise

**Chunks too small:**
- Each chunk loses surrounding context
- A single sentence may not carry enough meaning to embed well
- Retrieval may return fragments that don't make sense alone
- Answers may be incomplete or confusing

**The sweet spot** depends on your content type. A good starting point is 200–500 tokens per chunk with a small overlap of 20–50 tokens. You'll tune this based on how your retrieval performs.

| Chunk Size | Embedding Quality | Context Preserved | Retrieval Precision |
|------------|-------------------|-------------------|---------------------|
| Too large  | Blurry            | High (but noisy)  | Low                 |
| Too small  | Sharp but thin    | Low               | Medium              |
| Just right | Sharp             | Sufficient        | High                |

---

## Why It Matters

Chunking is the foundation of retrieval quality. Everything downstream — embeddings, similarity search, LLM generation — depends on having well-formed chunks.

- Better chunks → sharper embeddings
- Sharper embeddings → more accurate similarity search
- More accurate search → only relevant context reaches the LLM
- Relevant context → precise, trustworthy answers

A RAG system with poor chunking will underperform even with the best embedding model or the most powerful LLM. It's that foundational.

---

## Key Takeaways

- Large documents produce blurry embeddings — chunking fixes this
- Each chunk gets its own embedding and can be retrieved independently
- Fixed-size chunking is simple; semantic chunking is smarter
- Overlapping chunks prevent context loss at boundaries
- Chunking happens during ingestion, before anything hits the vector database
- Chunk size is a tunable parameter — too big and too small both hurt you
- Good chunking is one of the highest-leverage improvements in any RAG system

---

## What's Next

You now know how to split documents into retrievable pieces. But what happens when you have thousands of chunks across many different documents, topics, or time periods?

How do you make sure retrieval only pulls from the right subset?

That's where **metadata filtering** comes in.

On Day 10, we'll look at how attaching metadata to your chunks — things like source, date, category, or author — lets you filter before similarity search even runs. It's the difference between searching your entire library and searching only the shelf that matters.

---

*Day 9 of 21 — RAG Systems Guide*
*← [Day 8: Similarity Search](../day-08-similarity-search/README.md) | [Day 10: Metadata Filtering](../day-10-metadata-filtering/README.md) →*
