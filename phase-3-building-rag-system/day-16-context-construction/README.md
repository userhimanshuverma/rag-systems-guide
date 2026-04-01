# Day 16 — Context Construction

> "Retrieval finds the right information. Context construction makes it usable."

---

## The Problem

Your query pipeline just returned 5 relevant document chunks. Great.

But you can't just dump them raw into the LLM prompt. They might have duplicate content. They might be in the wrong order. Some might be too long. There's no structure telling the LLM where each piece came from.

Hand a messy pile of notes to someone and ask them to answer a question — they'll struggle. Organize those same notes clearly, label them, put the most important ones first — and suddenly the task is easy.

That's context construction. It's the step between retrieval and generation, and most people underestimate how much it matters.

> This is where good RAG becomes great RAG.

---

## What Context Construction Actually Is

Context construction is the process of taking raw retrieved chunks and transforming them into a clean, structured, token-aware context block that the LLM can reason over effectively.

It has four jobs:

1. **Clean** — normalize text, remove noise
2. **Deduplicate** — remove chunks with identical or near-identical content
3. **Order** — put the most relevant chunks first
4. **Format** — assemble into a structured block with source labels and token limits

The output is a single, well-organized string that gets inserted into the LLM prompt.

---

## The Analogy

Imagine you're about to answer a complex question and you have a stack of research notes.

Before you start writing, you:
- Throw out any duplicate notes
- Put the most relevant ones on top
- Label each note with where it came from
- Make sure you're not holding more notes than you can actually read

That's exactly what context construction does for your RAG system. The LLM is the writer. Context construction is the prep work that makes the writing good.

---

## Step-by-Step Flow

**Step 1 — Clean**
Normalize whitespace, strip stray characters, and standardize formatting across all chunks. Inconsistent formatting confuses the LLM and wastes tokens.

**Step 2 — Deduplicate**
Remove chunks with identical text. This happens more than you'd expect — the same content can appear in multiple documents or get retrieved twice from overlapping chunks. Duplicates waste token budget and can bias the LLM toward repeated information.

**Step 3 — Order by relevance**
Sort chunks by their relevance score, highest first. LLMs pay more attention to content at the beginning of the context window — so the most important information should come first.

**Step 4 — Build context with token budget**
Assemble chunks into a formatted string. Include source labels for traceability. Stop adding chunks once you hit the character/token budget — never overflow the context window.

---

## Architecture Flow

```
Retrieved Docs (from query pipeline)
        ↓
  Clean Text
(normalize whitespace, remove noise)
        ↓
  Deduplicate
(remove identical chunks)
        ↓
  Order by Relevance
(highest score first)
        ↓
  Build Context Block
(format with source labels + token budget)
        ↓
  LLM Prompt
(context + question → answer)
```

---

## The Code

No external dependencies needed — pure Python.

```python
import re


# ---------------------------------------------------------------------------
# Sample retrieved docs (output from Day 15 query pipeline)
# Already sorted by relevance score descending
# ---------------------------------------------------------------------------
retrieved_docs = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "metadata": {"source": "rag_intro.txt"}, "score": 0.91},
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "metadata": {"source": "rag_intro_v2.txt"}, "score": 0.90},  # duplicate
    {"text": "Vector databases enable fast semantic search over high-dimensional embeddings.", "metadata": {"source": "vector_db.txt"}, "score": 0.85},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "metadata": {"source": "embeddings.txt"}, "score": 0.78},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "metadata": {"source": "chunking.txt"}, "score": 0.72},
    {"text": "Reranking reorders retrieved documents by relevance before passing context to the LLM.", "metadata": {"source": "reranking.txt"}, "score": 0.65},
]


def clean_text(text: str) -> str:
    """Normalize whitespace and remove stray characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def deduplicate(docs: list[dict]) -> list[dict]:
    """
    Remove chunks with identical text content.
    Keeps the first occurrence (highest relevance since list is pre-sorted).
    """
    seen = set()
    unique = []
    for doc in docs:
        normalized = doc["text"].strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(doc)
    return unique


def order_by_relevance(docs: list[dict]) -> list[dict]:
    """Sort chunks by score descending — most relevant first."""
    return sorted(docs, key=lambda x: x.get("score", 0), reverse=True)


def build_context(docs: list[dict], max_chars: int = 1500) -> str:
    """
    Assemble retrieved chunks into a formatted context string.
    - Respects a character budget (proxy for token limit)
    - Includes source metadata for traceability
    - Stops adding chunks once budget is reached
    """
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = clean_text(doc["text"])
        source = doc.get("metadata", {}).get("source", "unknown")
        score = doc.get("score", "N/A")

        chunk_block = f"[{i}] Source: {source} (relevance: {score})\n{text}"
        chunk_len = len(chunk_block)

        if total_chars + chunk_len > max_chars:
            break

        context_parts.append(chunk_block)
        total_chars += chunk_len

    return "\n\n".join(context_parts)


def construct_context(docs: list[dict], max_chars: int = 1500) -> str:
    """
    Full pipeline: Clean → Deduplicate → Order → Build Context Block
    """
    for doc in docs:
        doc["text"] = clean_text(doc["text"])

    docs = deduplicate(docs)
    print(f"After deduplication: {len(docs)} chunks")

    docs = order_by_relevance(docs)
    print(f"Ordered by relevance score")

    context = build_context(docs, max_chars=max_chars)
    print(f"Context built: {len(context)} characters\n")

    return context


if __name__ == "__main__":
    print("=== Context Construction Pipeline ===\n")

    context = construct_context(retrieved_docs, max_chars=1500)

    print("--- Final Context Block ---\n")
    print(context)

    print("\n--- How this gets used in a prompt ---\n")
    query = "How does RAG improve LLM answers?"
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    print(prompt)
```

---

## Expected Output

```
=== Context Construction Pipeline ===

After deduplication: 5 chunks
Ordered by relevance score
Context built: 633 characters

--- Final Context Block ---

[1] Source: rag_intro.txt (relevance: 0.91)
RAG improves LLM accuracy by retrieving relevant documents before generating answers.

[2] Source: vector_db.txt (relevance: 0.85)
Vector databases enable fast semantic search over high-dimensional embeddings.

[3] Source: embeddings.txt (relevance: 0.78)
Embeddings convert text into numerical vectors that capture semantic meaning.

[4] Source: chunking.txt (relevance: 0.72)
Chunking splits large documents into smaller pieces for more precise retrieval.

[5] Source: reranking.txt (relevance: 0.65)
Reranking reorders retrieved documents by relevance before passing context to the LLM.

--- How this gets used in a prompt ---

You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
[1] Source: rag_intro.txt (relevance: 0.91)
RAG improves LLM accuracy by retrieving relevant documents before generating answers.
...

Question: How does RAG improve LLM answers?

Answer:
```

The duplicate chunk from `rag_intro_v2.txt` was removed. 6 chunks became 5. The most relevant chunk is first. The context is clean, labeled, and ready for the LLM.

---

## What Each Part Does

**`clean_text(text)`**
Collapses multiple spaces and newlines into single spaces. Chunks from PDFs and web pages often have messy whitespace — this normalizes it before assembly. Clean text = fewer wasted tokens.

**`deduplicate(docs)`**
Compares normalized (lowercased, stripped) text across all chunks. If two chunks have identical content, only the first is kept. Since the list is pre-sorted by relevance, the highest-scoring version survives. This is more common than you'd think — overlapping chunks and duplicate documents both cause this.

**`order_by_relevance(docs)`**
Sorts by score descending. This matters because LLMs have an attention bias — they process the beginning of the context more reliably than the middle. Put your best material first.

**`build_context(docs, max_chars)`**
The assembly function. Formats each chunk with a numbered label and source attribution, then concatenates them with a character budget. When the budget is hit, it stops — no chunk gets partially included. The `max_chars` parameter is a character-based proxy for token limits. In production you'd use a proper tokenizer to count tokens precisely.

**The final prompt**
Shows exactly how the context block slots into a real LLM prompt. The instruction tells the LLM to use only the provided context and to say "I don't know" if the answer isn't there — this is the key guardrail against hallucination.

---

## Important Considerations

**Token limits are real**
Every LLM has a maximum context window. If your context block is too large, it gets truncated — silently, in most cases. Always budget your context size. A rough rule: leave at least 30% of the context window for the question, system prompt, and the LLM's response.

**Order matters more than you think**
Research consistently shows LLMs perform better when the most relevant content is at the start of the context. Don't randomize chunk order. Always sort by relevance.

**Deduplication prevents bias**
If the same sentence appears three times in your context, the LLM will weight it more heavily — even if it's not the most important piece. Deduplication keeps the context balanced.

**Source labels enable trust and debugging**
Including `[1] Source: filename.txt` in each chunk block lets you trace every part of the LLM's answer back to a specific document. This is invaluable for debugging wrong answers and for building user-facing citation features.

**"I don't know" is a feature, not a failure**
Instructing the LLM to say "I don't know" when the context doesn't contain the answer is one of the most important guardrails in RAG. Without it, the model will hallucinate to fill the gap.

---

## Key Takeaways

- Context construction transforms raw retrieved chunks into a clean, structured LLM input
- Four stages: clean → deduplicate → order by relevance → build with token budget
- Deduplication removes identical chunks that waste tokens and bias the LLM
- Most relevant chunks go first — LLMs pay more attention to the start of the context
- Always enforce a character/token budget — never overflow the context window
- Source labels in the context block enable traceability and citation
- The "I don't know" instruction is a critical hallucination guardrail
- This step is often skipped in demos but is essential in production systems

---

## What's Next

The context block is ready. Now it needs to be combined with the right prompt to actually get a good answer from the LLM.

On **Day 17**, we'll cover Prompting with Retrieved Knowledge — how to structure the system prompt, how to instruct the LLM to use context correctly, and the prompting patterns that separate reliable RAG systems from unreliable ones.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
