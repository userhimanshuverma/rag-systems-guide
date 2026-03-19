# Day 3 — RAG vs Fine-Tuning: When to Retrieve vs When to Train

> "Use the right tool for the right job. A hammer is great — but not for every nail."

---

## The Confusion

Here's a question that comes up constantly when people start building AI systems:

*"Should I use RAG or just fine-tune the model on my data?"*

It's a fair question. Both approaches seem to solve the same problem — making the model more useful for your specific use case. But they're actually solving very different problems, and mixing them up leads to wasted time, money, and frustration.

The confusion exists because both RAG and fine-tuning involve "teaching" the model something new. But what they teach — and how — is completely different.

Let's clear this up once and for all.

---

## The Simple Explanation

Here's the one-line version of each:

> **RAG** = Give the model the right information at the time it needs to answer.

> **Fine-Tuning** = Change how the model thinks, speaks, or behaves.

RAG is about **knowledge** — what the model knows.
Fine-tuning is about **behavior** — how the model acts.

These are two separate dimensions. And that distinction is everything.

---

## The Core Difference

| | RAG | Fine-Tuning |
|---|---|---|
| What it changes | The information available at query time | The model's weights and behavior |
| When it applies | At runtime, dynamically | At training time, statically |
| Best for | Factual knowledge, private data, real-time info | Tone, style, format, task-specific behavior |
| Update cost | Just update your documents | Retrain the model (expensive) |
| Risk of hallucination | Lower (grounded in real docs) | Higher (model may still guess) |
| Flexibility | High — swap docs anytime | Low — changes are baked in |

The key mental model:

- **RAG changes what the model knows**
- **Fine-tuning changes who the model is**

---

## The Analogy That Makes It Click

Imagine you hire a new employee.

**Fine-tuning** is like putting them through a training program. You teach them your company's communication style, how to write emails, how to handle customer calls, what tone to use. After training, they naturally behave the way you want — without needing to be told every time.

**RAG** is like giving that same employee access to a shared drive full of company documents, policies, and data. When a customer asks a specific question, they don't need to have memorized the answer — they just look it up and respond accurately.

Now here's the key insight:

Training the employee doesn't help if they don't have access to the right documents. And giving them documents doesn't help if they don't know how to communicate professionally.

**The best systems use both.** But you need to know which problem you're actually solving first.

---

## When to Use RAG

RAG is the right choice when your problem is about **knowledge** — specifically, knowledge that:

**Changes frequently**
Product prices, policies, news, inventory — anything that updates regularly. You can't retrain a model every time a document changes. With RAG, just update the knowledge base.

**Is private or internal**
Your company's internal docs, customer records, legal files, codebases — this data was never in the model's training set and shouldn't be. RAG lets you use it without exposing it during training.

**Needs to be traceable**
When users need to know *where* an answer came from, RAG lets you point to the exact source document. Fine-tuning bakes knowledge in invisibly.

**Is too large to fit in training**
You have thousands of documents. You can't fine-tune on all of them efficiently. RAG retrieves only what's relevant, on demand.

**Real-world examples:**
- Customer support bot that answers from your help docs
- Internal Q&A tool over company policies
- Research assistant that searches through papers
- Legal assistant that references case files

---

## When to Use Fine-Tuning

Fine-tuning is the right choice when your problem is about **behavior** — specifically:

**Tone and style**
You want the model to always respond in a specific voice — formal, casual, technical, empathetic. Fine-tuning bakes this in so you don't have to prompt for it every time.

**Output format**
You always want JSON output, or a specific report structure, or responses under 100 words. Fine-tuning makes this the default behavior.

**Domain-specific reasoning**
Medical diagnosis patterns, legal reasoning structures, code review conventions — these are thinking patterns, not facts. Fine-tuning teaches the model *how* to think in your domain.

**Reducing prompt complexity**
If you're writing massive system prompts to get consistent behavior, fine-tuning can replace that with trained behavior.

**Real-world examples:**
- A model that always responds in your brand's voice
- A coding assistant that follows your team's conventions
- A classifier that categorizes support tickets in a specific way
- A model that always outputs structured data in a fixed schema

---

## Common Mistakes

**Mistake 1: Using fine-tuning to inject knowledge**
This is the most common one. People fine-tune a model on their company's FAQ, hoping it'll "learn" the answers. It might — but the knowledge gets baked in statically. When the FAQ changes, you have to retrain. And the model may still hallucinate on edge cases. Use RAG for this instead.

**Mistake 2: Using RAG when you need behavior change**
If your problem is that the model doesn't respond in the right format or tone, adding more documents won't fix it. That's a behavior problem — fine-tuning territory.

**Mistake 3: Assuming fine-tuning eliminates hallucinations**
It doesn't. Fine-tuning changes behavior, not reliability. A fine-tuned model can still confidently make things up. RAG reduces hallucinations by grounding answers in real documents.

**Mistake 4: Overcomplicating early**
Most teams jump to fine-tuning before they've even tried RAG with a good prompt. Start simple. RAG + a well-crafted prompt solves 80% of real-world problems without the cost and complexity of fine-tuning.

---

## The Combined Approach

In production systems, RAG and fine-tuning aren't competitors — they're complements.

A well-designed system might look like this:

- Fine-tune the model to always respond in a specific format, follow your brand tone, and handle edge cases gracefully
- Use RAG to give that fine-tuned model access to fresh, private, up-to-date knowledge at query time

The fine-tuned model handles *how* it responds. RAG handles *what* it knows.

---

## Architecture Thinking

```
User Query
     ↓
  Retriever
(fetches relevant documents from knowledge base)
     ↓
Retrieved Context
(real, specific, up-to-date information)
     ↓
Fine-Tuned LLM
(knows how to respond in your style and format)
     ↓
Final Answer
(accurate knowledge + consistent behavior)
```

Each layer does one job. RAG handles knowledge. Fine-tuning handles behavior. Together, they cover both dimensions.

---

## Key Takeaways

- RAG and fine-tuning solve different problems — don't confuse them
- RAG = knowledge at runtime (dynamic, flexible, traceable)
- Fine-tuning = behavior at training time (static, baked-in, expensive to change)
- Use RAG when your data changes, is private, or needs to be sourced
- Use fine-tuning when you need consistent tone, format, or reasoning patterns
- Fine-tuning does NOT reliably inject factual knowledge — use RAG for that
- Most production systems use both, but start with RAG — it solves more problems with less cost
- Always ask: "Is this a knowledge problem or a behavior problem?" before deciding

---

## What's Next

Now that you know *what* RAG is and *when* to use it, it's time to look at the actual pipeline.

On **Day 4**, we'll walk through the Basic RAG Pipeline end to end — from the moment a user types a question to the moment an answer comes back. Every step, explained simply.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
