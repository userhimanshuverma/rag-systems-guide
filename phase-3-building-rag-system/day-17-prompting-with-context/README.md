# Day 17 — Prompting with Retrieved Knowledge

> "Retrieval gives the LLM the right information. The prompt tells it what to do with that information."

---

## The Problem

You've retrieved the right documents. You've built a clean context block. You pass it to the LLM.

And the answer is still wrong.

Not because the retrieval failed. Not because the context was bad. But because the LLM didn't know how to use it. It mixed in outside knowledge. It ignored parts of the context. It made up details that weren't there.

This is a prompting problem — and it's more common than people expect.

The prompt is the instruction layer. It tells the LLM what role to play, how to use the context, what to do when it doesn't know something, and how to format its response. Get the prompt wrong and even perfect retrieval produces bad answers.

> RAG is not just about data. It's about control.

---

## What the Prompt Actually Does

In a RAG system, the prompt has three jobs:

1. **Set the role** — tell the LLM what kind of assistant it is
2. **Constrain the behavior** — tell it to use only the provided context
3. **Handle the unknown** — tell it what to say when the context doesn't have the answer

Without these three things, the LLM will do what it always does — generate the most plausible-sounding response from its training data. That's exactly what you're trying to avoid.

---

## Good Prompt vs Bad Prompt

**Bad prompt:**
```
Answer this question: How does RAG work?
```
The LLM answers from memory. No context used. Hallucination risk is high.

**Also bad:**
```
Here is some context. Answer the question.
[context]
How does RAG work?
```
Vague instruction. The LLM might use the context, might not. No guardrail for unknown answers.

**Good prompt:**
```
You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
[context]

Question: How does RAG work?

Answer:
```

Explicit role. Clear constraint. Defined fallback. This is the difference between a demo and a production system.

---

## The Analogy

Imagine you hand someone a stack of research notes and ask them to answer a question.

Without instructions, they might answer from memory and ignore your notes entirely.

With clear instructions — *"Use only these notes. If the answer isn't here, say so."* — they stay grounded. They reference the material. They don't make things up.

The prompt is those instructions. The context is the notes. The LLM is the person answering.

---

## Prompt Structure

Every RAG prompt has three parts:

```
[System Instruction]
Role + behavior rules + fallback instruction

[Context Block]
Retrieved chunks from Day 16

[User Question]
The actual query
```

This structure is consistent across all prompt patterns. What changes is how strict the instructions are and what format the answer should take.

---

## Architecture Flow

```
Retrieved Context (Day 16)
        +
Prompt Instructions (role, rules, fallback)
        +
User Question
        ↓
     LLM
        ↓
Grounded Answer
```

---

## The Code — 4 Prompt Patterns

```python
"""
Day 17 — Prompting with Retrieved Knowledge

Demonstrates different prompt patterns for RAG systems:
- Basic prompt
- Strict grounding prompt (no hallucination)
- Summarization prompt
- Conversational prompt with history
"""

# Sample context (output from Day 16 context construction)
sample_context = """[1] Source: rag_intro.txt (relevance: 0.91)
RAG improves LLM accuracy by retrieving relevant documents before generating answers.

[2] Source: vector_db.txt (relevance: 0.85)
Vector databases enable fast semantic search over high-dimensional embeddings.

[3] Source: embeddings.txt (relevance: 0.78)
Embeddings convert text into numerical vectors that capture semantic meaning."""


# ---------------------------------------------------------------------------
# Pattern 1 — Basic RAG Prompt
# Simple, works for most use cases
# ---------------------------------------------------------------------------

def build_basic_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Pattern 2 — Strict Grounding Prompt
# Forces citation — best for legal, medical, compliance use cases
# ---------------------------------------------------------------------------

def build_strict_prompt(context: str, question: str) -> str:
    return f"""You are a precise assistant. Follow these rules strictly:
1. Answer ONLY using the context below. Do not use any outside knowledge.
2. For every statement in your answer, cite the source number in brackets e.g. [1].
3. If the context does not contain enough information, respond with:
   "The provided context does not contain sufficient information to answer this question."
4. Do not speculate, infer, or add information not explicitly stated in the context.

Context:
{context}

Question: {question}

Answer (with citations):"""


# ---------------------------------------------------------------------------
# Pattern 3 — Summarization Prompt
# When you want synthesis, not a direct answer
# ---------------------------------------------------------------------------

def build_summary_prompt(context: str, topic: str) -> str:
    return f"""You are a helpful assistant that summarizes technical content clearly.
Using ONLY the context below, write a concise summary about: {topic}

Rules:
- Keep the summary under 100 words
- Use plain language
- Do not add information not present in the context

Context:
{context}

Summary:"""


# ---------------------------------------------------------------------------
# Pattern 4 — Conversational Prompt with Chat History
# For multi-turn RAG chatbots
# ---------------------------------------------------------------------------

def build_conversational_prompt(context: str, question: str, history: list[dict]) -> str:
    """
    history format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    history_text = ""
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    return f"""You are a helpful assistant in an ongoing conversation.
Answer using ONLY the context provided. If the answer is not in the context, say "I don't know."

Context:
{context}

Conversation so far:
{history_text.strip()}

User: {question}
Assistant:"""


# ---------------------------------------------------------------------------
# Run all patterns
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


if __name__ == "__main__":
    question = "How does RAG improve LLM answers?"

    separator("Pattern 1: Basic RAG Prompt")
    print(build_basic_prompt(sample_context, question))

    separator("Pattern 2: Strict Grounding Prompt")
    print(build_strict_prompt(sample_context, question))

    separator("Pattern 3: Summarization Prompt")
    print(build_summary_prompt(sample_context, "how RAG systems work"))

    separator("Pattern 4: Conversational Prompt")
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating an answer."},
    ]
    print(build_conversational_prompt(sample_context, "And how does it reduce hallucinations?", history))
```

---

## Expected Output

```
============================================================
  Pattern 1: Basic RAG Prompt
============================================================
You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
[1] Source: rag_intro.txt (relevance: 0.91)
RAG improves LLM accuracy by retrieving relevant documents before generating answers.
...

Question: How does RAG improve LLM answers?

Answer:

============================================================
  Pattern 2: Strict Grounding Prompt
============================================================
You are a precise assistant. Follow these rules strictly:
1. Answer ONLY using the context below. Do not use any outside knowledge.
2. For every statement in your answer, cite the source number in brackets e.g. [1].
...

============================================================
  Pattern 4: Conversational Prompt
============================================================
...
User: What is RAG?
Assistant: RAG stands for Retrieval Augmented Generation...

User: And how does it reduce hallucinations?
Assistant:
```

---

## What Each Pattern Does

**Pattern 1 — Basic**
The go-to for most use cases. Clear role, clear constraint, clear fallback. Simple and effective. Start here.

**Pattern 2 — Strict with Citations**
Forces the LLM to cite source numbers for every claim. Maximally reduces hallucination. Use this for high-stakes domains — legal, medical, compliance, finance — where every statement needs to be traceable.

**Pattern 3 — Summarization**
When you don't want a direct Q&A answer but a synthesized summary of the context. Useful for document summarization, briefings, and digest-style outputs.

**Pattern 4 — Conversational**
Includes chat history so the LLM can maintain context across multiple turns. Essential for chatbot-style RAG applications where users ask follow-up questions.

---

## Why Instructions Matter

**"Use ONLY the context"** — without this, the LLM blends retrieved content with training memory. You lose control over what the answer is based on.

**"If not in context, say I don't know"** — without this, the LLM fills gaps with hallucinated content. This single instruction is one of the most important guardrails in any RAG system.

**Numbered citations** — forcing `[1]`, `[2]` style citations makes every claim traceable. You can audit answers, debug wrong responses, and show users exactly where information came from.

**Explicit role** — "You are a helpful assistant" sets the tone and behavior. More specific roles ("You are a customer support agent for Acme Corp") constrain the LLM further and improve consistency.

---

## Best Practices

**Be explicit, not implicit**
Don't assume the LLM will figure out what you want. State it directly. "Use only the context" is better than "use the context."

**Always define the fallback**
What should the LLM say when it doesn't know? Define it. "I don't know" is better than a hallucinated answer.

**Keep instructions at the top**
LLMs pay more attention to instructions at the beginning of the prompt. Put your rules before the context, not after.

**Match prompt strictness to use case**
A casual Q&A chatbot can use Pattern 1. A medical information system should use Pattern 2. Don't over-engineer simple use cases, but don't under-engineer critical ones.

**Test with adversarial questions**
Ask questions that aren't in the context. Ask ambiguous questions. Ask questions that partially overlap with the context. This is how you find prompt weaknesses before users do.

---

## Key Takeaways

- The prompt is the instruction layer — it tells the LLM how to use the retrieved context
- Without explicit constraints, LLMs blend context with training memory and hallucinate
- Every RAG prompt needs three things: role, constraint ("use only context"), and fallback ("say I don't know")
- Four patterns cover most use cases: basic, strict with citations, summarization, conversational
- Forced citations make answers traceable and auditable — essential for high-stakes domains
- Instructions at the top of the prompt get more attention from the LLM
- Test with questions that aren't in the context — that's where prompt weaknesses show up
- RAG is not just about retrieval. The prompt is where you take control of the output.

---

## What's Next

Phase 3 is complete. You've built the full RAG pipeline from ingestion to prompting.

Phase 4 starts on **Day 18** with Multi-Query Retrieval — what happens when a single query isn't enough to capture everything the user is asking, and how generating multiple search queries dramatically improves recall.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
