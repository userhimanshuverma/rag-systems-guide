# Day 4 — The Basic RAG Pipeline

> "A system is only as strong as its weakest step. Know every step."

---

## The Question

By now you know what RAG is and why it exists.

But knowing the concept and understanding the actual flow are two different things. When you sit down to build a RAG system — or debug one — you need to know exactly what happens at each step, in what order, and why.

So today we walk through the full pipeline. Every step. No hand-waving.

---

## The Pipeline at a Glance

The basic RAG pipeline has five stages:

```
User Query
    ↓
Retriever
(searches the knowledge base)
    ↓
Relevant Documents
(the right chunks of information)
    ↓
LLM
(reads context + question, generates answer)
    ↓
Final Answer
```

Simple on the surface. But each step has real depth — and real failure modes. Let's go through them one by one.

---

## Step-by-Step Breakdown

### Step 1 — User Query

Everything starts here. A user types a question or sends a request.

This could be:
- "What is our return policy?"
- "Summarize the Q3 earnings report"
- "How do I reset my password?"

The query is the raw input. It drives everything that follows.

One thing worth noting early: **the quality of the query matters**. A vague or ambiguous question makes it harder for the retriever to find the right documents. We'll explore this more in later days when we cover query expansion and multi-query retrieval.

---

### Step 2 — The Retriever

This is the search engine of your RAG system.

The retriever takes the user's query and searches through your knowledge base to find the most relevant pieces of information. It doesn't just do keyword matching — it understands meaning. (We'll cover how this works with embeddings on Day 6.)

The retriever returns a ranked list of document chunks — the pieces of text most likely to contain the answer.

Key things the retriever does:
- Converts the query into a searchable format
- Compares it against stored documents
- Returns the top N most relevant chunks

The retriever is arguably the most critical component. If it returns the wrong documents, the LLM has nothing useful to work with — and the answer will be wrong no matter how good the model is.

---

### Step 3 — Retrieved Documents

The retriever hands back a set of document chunks — small pieces of text pulled from your knowledge base.

These chunks might come from:
- A PDF manual
- A support article
- A database record
- An internal wiki page
- A code file

They're not full documents — they're targeted excerpts. Just the relevant parts.

These chunks get assembled into a context block that will be passed to the LLM alongside the original question.

This step is where **chunking strategy** matters a lot. If your documents are split poorly — too large, too small, or at the wrong boundaries — the retrieved context will be messy. We'll cover chunking in depth on Day 9.

---

### Step 4 — The LLM

Now the LLM enters the picture.

It receives two things:
1. The original user question
2. The retrieved context (the document chunks)

The prompt looks something like this conceptually:

```
Here is some relevant information:
[retrieved document chunks]

Using only the information above, answer the following question:
[user question]
```

The LLM reads both, reasons over the context, and generates a response. It's not guessing from memory anymore — it's working from real, specific information you gave it.

This is why RAG reduces hallucinations. The model has actual content to reference. It doesn't need to invent an answer.

---

### Step 5 — Final Answer

The LLM's output is returned to the user.

In a well-built system, the answer is:
- Grounded in the retrieved documents
- Specific to the user's question
- Traceable back to a source

That last point — traceability — is one of RAG's biggest practical advantages. You can show users exactly which document the answer came from. That builds trust and makes debugging much easier.

---

## The Analogy

Think about how a good researcher works.

They get a question. They don't just answer from memory. They:

1. Search for relevant sources
2. Read the most relevant parts
3. Synthesize an answer based on what they found

That's exactly the RAG pipeline.

- **Search** = Retriever
- **Read** = Retrieved Documents passed as context
- **Write** = LLM generating the final answer

The LLM is the writer. But a writer is only as good as their research.

---

## How the Components Connect

```
┌─────────────────────────────────────────────┐
│                  RAG Pipeline               │
│                                             │
│  User Query                                 │
│       │                                     │
│       ▼                                     │
│  ┌─────────┐      ┌──────────────────────┐  │
│  │Retriever│ ───► │   Knowledge Base     │  │
│  └─────────┘      │ (docs, PDFs, wikis)  │  │
│       │           └──────────────────────┘  │
│       ▼                                     │
│  Retrieved Chunks                           │
│       │                                     │
│       ▼                                     │
│  ┌──────────────────────────────────────┐   │
│  │  LLM                                 │   │
│  │  Input: [context] + [user question]  │   │
│  └──────────────────────────────────────┘   │
│       │                                     │
│       ▼                                     │
│  Final Answer                               │
└─────────────────────────────────────────────┘
```

Each component has one job. They're loosely coupled — you can swap out the retriever, change the knowledge base, or upgrade the LLM independently.

---

## Where Things Can Go Wrong

The pipeline looks clean. But each step is a potential failure point.

**Bad retrieval**
The retriever returns documents that aren't actually relevant to the question. The LLM then generates an answer based on the wrong context — and it'll sound confident doing it.

**Poor chunking**
If documents are split at the wrong places, retrieved chunks might be missing key context. Imagine retrieving the middle of a paragraph that only makes sense with the sentence before it.

**Too much context**
Dumping too many chunks into the prompt overwhelms the LLM. It may miss the most relevant part, or the prompt exceeds the model's token limit entirely.

**Too little context**
Retrieving only one chunk when the answer spans multiple sections means the LLM gets an incomplete picture.

**Irrelevant knowledge base**
If the documents in your knowledge base don't contain the answer, no amount of good retrieval will help. Garbage in, garbage out.

**Prompt construction issues**
How you assemble the context and question into a prompt matters. A poorly structured prompt leads to poorly structured answers even with perfect retrieval.

---

## The Key Insight

Here it is, and it's worth repeating throughout this series:

> **The LLM is only as good as the context it receives.**

The model itself isn't the bottleneck in most RAG systems. The retrieval is. If you give the LLM the right information, it will almost always generate a good answer. If you give it the wrong information — or no information — even the best model will fail.

This means most of your engineering effort in a RAG system should go into the retrieval layer, not the generation layer.

---

## Key Takeaways

- The basic RAG pipeline has five steps: Query → Retriever → Documents → LLM → Answer
- The retriever is the most critical component — bad retrieval breaks everything downstream
- Retrieved chunks are assembled into context and passed to the LLM alongside the question
- The LLM generates answers grounded in real documents, not memory
- Each step is a potential failure point — understanding them is how you debug and improve
- The LLM is only as good as the context it receives — invest in retrieval quality
- Components are loosely coupled — you can improve each one independently

---

## What's Next

Now that you understand how the pipeline works, the next question is: where does it break?

On **Day 5**, we'll go deep into the failure modes of basic RAG — bad retrieval, irrelevant context, token limits, and hallucinations that still slip through. Understanding these failures is what separates engineers who build RAG systems from engineers who build *reliable* RAG systems.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
