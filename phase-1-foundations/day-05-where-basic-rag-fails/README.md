# Day 5 — Where Basic RAG Fails

> "A system that looks simple on paper can break in a dozen ways in production."

---

## The Hook

You've built your first RAG system. The pipeline is clean. The components are connected. You test it — and it works.

Then a real user asks a real question. And the answer is confidently, completely wrong.

This isn't rare. It's one of the most common experiences engineers have when they first deploy RAG. The system looks right. The architecture is correct. But the answers aren't reliable.

Why?

Because RAG has a deceptively simple surface and a surprisingly fragile interior. The pipeline depends on every step working well — and there are more ways for each step to go wrong than most people expect.

Today we map out exactly where basic RAG breaks down.

---

## The Core Dependency

Before we get into failure modes, remember this from Day 4:

> The LLM is only as good as the context it receives.

This means every failure in a RAG system traces back to one of two things:

1. The retriever returned the wrong information
2. The context passed to the LLM was poorly constructed

The LLM itself is rarely the problem. The pipeline feeding it is.

---

## Failure Point 1 — Bad Retrieval

This is the most common and most damaging failure.

The retriever searches your knowledge base and returns documents — but the wrong ones. They might be topically related but not actually relevant. Or they might be from the right document but the wrong section.

**Why it happens:**
- The query is vague or ambiguous
- The knowledge base isn't well organized
- The search method doesn't understand meaning well enough
- The documents weren't indexed properly

**What the LLM does with it:**
It tries its best. It reads the wrong documents and generates an answer based on them. The answer sounds confident and well-structured — but it's built on the wrong foundation.

This is the RAG equivalent of a researcher citing the wrong paper. The writing looks great. The source is wrong.

---

## Failure Point 2 — Irrelevant Context (Too Much Noise)

More retrieved documents doesn't mean better answers.

When the retriever returns too many chunks — or chunks that are only loosely related — the LLM gets flooded with noise. It has to sift through irrelevant information to find what actually matters.

**Why it happens:**
- Retrieval threshold is set too low (returns everything above a minimal score)
- Knowledge base has duplicate or overlapping content
- Chunks are too large and contain mixed topics

**What the LLM does with it:**
It gets confused. It may focus on the wrong part of the context, blend information from unrelated chunks, or produce a vague answer that tries to cover too much ground.

Signal-to-noise ratio matters. A single highly relevant chunk beats five loosely relevant ones.

---

## Failure Point 3 — Token Limits

LLMs have a maximum context window — a limit on how much text they can process at once.

If your retrieved documents are too long, or you retrieve too many chunks, you'll hit this limit. And when you do, something gets cut off — usually silently.

**Why it happens:**
- Documents aren't chunked into small enough pieces
- Too many chunks are retrieved and concatenated
- The system doesn't account for the space taken by the prompt itself

**What the LLM does with it:**
It only sees part of the context. The answer it generates may be missing key information that was cut off. Worse, you might not even know this is happening unless you're monitoring token counts.

**The tricky part:**
Even within the token limit, LLMs tend to pay more attention to content at the beginning and end of the context window. Information buried in the middle can get underweighted. This is sometimes called the "lost in the middle" problem.

---

## Failure Point 4 — Hallucination Still Happens

RAG reduces hallucinations — but it doesn't eliminate them.

Even with retrieved context, an LLM can still generate incorrect information. This happens more than people expect.

**Why it happens:**
- The retrieved context doesn't fully answer the question, so the model fills in the gaps from memory
- The model misreads or misinterprets the context
- The question requires reasoning across multiple documents and the model makes a logical error
- The model is instructed to always give an answer, even when it shouldn't

**What it looks like:**
The answer contains a mix of real retrieved information and invented details. The invented parts blend in seamlessly. This is harder to catch than a fully hallucinated answer.

---

## Real-World Examples

**Example 1 — Customer Support Bot**
A user asks: "What's the cancellation policy for enterprise plans?"
The retriever returns chunks about the general cancellation policy — but misses the enterprise-specific section buried in a different document. The LLM answers confidently using the wrong policy. The user gets incorrect information.

**Example 2 — Internal Knowledge Base**
An employee asks: "What's the process for requesting hardware?"
The knowledge base has an outdated version of the process and a newer one. The retriever returns both. The LLM blends them into a hybrid answer that doesn't match either actual process.

**Example 3 — Research Assistant**
A researcher asks a complex multi-part question. The retriever returns relevant chunks, but the answer requires connecting information across three different documents. The LLM only reasons over the top two chunks and misses the third. The answer is incomplete.

**Example 4 — Token Overflow**
A user asks a broad question. The system retrieves 10 chunks. Together with the system prompt, they exceed the token limit. The last three chunks get silently cut. The LLM answers without the most recent information — which happened to be in those last chunks.

---

## The Analogy

Imagine you're about to take an open-book exam.

You're allowed to bring notes. You spend the night preparing — but you grab the wrong notebook. Or you grab the right one but it's so disorganized you can't find the relevant page. Or the notebook is so thick that you only have time to read the first half.

You sit down, look at your notes, and write your best answer. It sounds confident. But it's based on the wrong material.

That's basic RAG failing. The exam (LLM) isn't the problem. The notes (retrieval + context) are.

---

## How Failure Happens in the Pipeline

Here's what a broken RAG flow looks like:

```
User Question
      ↓
  Retriever
(returns wrong or noisy documents)
      ↓
Wrong / Irrelevant Documents
(bad context assembled for LLM)
      ↓
     LLM
(reads bad context, generates answer anyway)
      ↓
Confident but Wrong Answer
(user has no idea it's wrong)
```

Compare this to the happy path from Day 4. The structure is identical. The failure is invisible from the outside — which is what makes it dangerous.

---

## The Key Insight

> Basic RAG is easy to build. Reliable RAG is hard to build.

The gap between a demo that works and a system that works consistently in production is almost entirely about retrieval quality and context construction.

Most RAG failures aren't LLM failures. They're retrieval failures. And most retrieval failures come down to:
- How documents are stored and indexed
- How queries are processed before retrieval
- How chunks are sized and structured
- How many results are retrieved and how they're filtered

Fix the retrieval, and most of the other problems follow.

---

## How to Think About Fixing It (High Level)

We'll go deep on all of these in later days, but here's the direction:

**Better retrieval**
Use semantic search (embeddings) instead of keyword matching. Understand meaning, not just words. We start this on Day 6.

**Cleaner context**
Filter out low-relevance chunks. Rerank results to put the best ones first. Don't just dump everything into the prompt.

**Better chunking**
Split documents at meaningful boundaries. Make chunks the right size — not too large, not too small. Preserve context across chunk boundaries. Day 9 covers this in full.

**Smarter query handling**
Rephrase or expand the user's query before retrieval to improve search results. Day 18 covers multi-query retrieval.

**Guardrails on generation**
Instruct the LLM to say "I don't know" when the context doesn't contain the answer, rather than guessing.

---

## Key Takeaways

- Basic RAG fails in four main ways: bad retrieval, noisy context, token limits, and residual hallucination
- Bad retrieval is the most common and most damaging failure — it poisons everything downstream
- More retrieved documents doesn't mean better answers — noise hurts as much as missing information
- Token limits are a real constraint — context gets cut silently if you're not careful
- RAG reduces hallucinations but doesn't eliminate them — the model can still fill gaps with invented content
- Most RAG failures are retrieval failures, not LLM failures
- The fix starts with better embeddings, smarter chunking, and cleaner context construction

---

## What's Next

We've now seen the full picture of basic RAG — what it is, how it works, and where it breaks.

Phase 2 starts on **Day 6**, and it's all about fixing the retrieval layer from the ground up.

We start with **embeddings** — the technology that makes semantic search possible. Understanding embeddings is the foundation of everything that comes next: vector databases, similarity search, chunking, and hybrid retrieval.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
