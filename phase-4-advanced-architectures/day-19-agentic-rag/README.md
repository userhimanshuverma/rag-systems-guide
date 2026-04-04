# Day 19 — Agentic RAG

> "A fixed pipeline always does the same thing. An agent does what the situation requires."

---

## The Problem

Basic RAG is a linear pipeline. Query goes in, chunks come out, LLM generates an answer. Every query follows the same path, regardless of how complex or ambiguous it is.

That works fine for simple, well-formed questions. But real users ask messy questions.

- *"Explain how everything connects in a RAG system"* — too broad for a single query
- *"Why is my retrieval returning wrong results?"* — needs multiple angles to answer well
- *"Compare RAG and fine-tuning for my use case"* — requires reasoning, not just retrieval

A fixed pipeline doesn't adapt. It retrieves once, passes whatever it got to the LLM, and hopes for the best. If the first retrieval was weak, the answer is weak — and the system has no way to recover.

Agentic RAG fixes this by replacing the fixed pipeline with a decision loop.

---

## What Agentic RAG Actually Is

In standard RAG, the flow is predetermined:

```
Query → Retrieve → Generate → Done
```

In Agentic RAG, the flow is dynamic:

```
Query → Retrieve → Evaluate → Decide → (Retrieve again? Refine? Answer? Stop?)
```

An agent sits in the middle and makes decisions at each step. It looks at what was retrieved, judges whether it's good enough, and decides what to do next — retrieve more, refine the query, or generate the answer.

> The agent doesn't follow a script. It responds to what it finds.

---

## The Key Idea: Decision Loop

The core difference between standard RAG and Agentic RAG is the loop.

Standard RAG runs once. Agentic RAG runs until it's satisfied — or until it hits a step limit.

At each iteration, the agent asks:
- Are these results good enough to answer the question?
- If yes → generate the answer
- If no → what should I do differently?
  - Refine the query (add more specificity)
  - Broaden the query (remove constraints)
  - Try a different angle entirely
  - Give up if max steps reached

This loop is what makes the system adaptive. It can recover from bad first retrievals, handle ambiguous queries, and accumulate context across multiple retrieval rounds.

---

## Architecture Flow

```
User Query
    ↓
  Agent
    ↓
Retrieve Documents
    ↓
Evaluate Results
(are they sufficient?)
    ↙         ↓         ↘
Refine     Broaden      Answer
Query      Query          ↓
  ↓          ↓        Generate
  └──────────┘        Final Answer
       ↓
  Retrieve Again
  (next iteration)
       ↓
  Evaluate Again
       ↓
  ... (up to max_steps)
       ↓
  Stop (use best available)
```

---

## Step-by-Step Flow

**Step 1 — Query arrives**
The agent receives the user's question and starts the loop.

**Step 2 — Retrieve**
Run semantic search for the current query. Get top K results.

**Step 3 — Evaluate**
Score the results. Are enough of them above the relevance threshold? Is there sufficient coverage?

**Step 4 — Decide**
Based on evaluation:
- `answer` — results are good, generate the answer
- `refine` — results are weak, add more specificity to the query
- `broaden` — results are empty, simplify the query
- `stop` — max steps reached, use whatever is available

**Step 5 — Iterate or answer**
If the decision is to continue, update the query and go back to Step 2. Otherwise, generate the final answer from accumulated results.

---

## The Code

```python
import numpy as np
from sentence_transformers import SentenceTransformer

knowledge_base = [
    {"text": "RAG improves LLM accuracy by retrieving relevant documents before generating answers.", "source": "rag_intro.txt"},
    {"text": "Vector databases store embeddings and enable fast nearest-neighbor search at scale.", "source": "vector_db.txt"},
    {"text": "Embeddings convert text into numerical vectors that capture semantic meaning.", "source": "embeddings.txt"},
    {"text": "Chunking splits large documents into smaller pieces for more precise retrieval.", "source": "chunking.txt"},
    {"text": "Reranking reorders retrieved documents by relevance score before passing to the LLM.", "source": "reranking.txt"},
    {"text": "Hallucinations occur when LLMs generate confident but factually incorrect answers.", "source": "hallucinations.txt"},
    {"text": "Fine-tuning adjusts model weights to change behavior, not to inject new knowledge.", "source": "finetuning.txt"},
    {"text": "Metadata filtering narrows the search space before similarity search runs.", "source": "metadata.txt"},
    {"text": "Hybrid search combines keyword search and semantic search for better recall.", "source": "hybrid.txt"},
    {"text": "Context construction organizes retrieved chunks into a clean, token-aware prompt block.", "source": "context.txt"},
    {"text": "Agentic RAG uses a decision loop to dynamically decide when and what to retrieve.", "source": "agentic.txt"},
    {"text": "Multi-query retrieval generates multiple query variations to improve recall.", "source": "multi_query.txt"},
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

for doc in knowledge_base:
    doc["embedding"] = model.encode(doc["text"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    query_vec = model.encode(query)
    scored = []
    for doc in docs:
        score = cosine_similarity(query_vec, doc["embedding"])
        scored.append({"text": doc["text"], "source": doc["source"], "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


RELEVANCE_THRESHOLD = 0.45
COVERAGE_THRESHOLD = 2


def evaluate_results(results: list[dict]) -> dict:
    if not results:
        return {"status": "empty", "good_count": 0, "best_score": 0.0}
    good_results = [r for r in results if r["score"] >= RELEVANCE_THRESHOLD]
    return {
        "status": "sufficient" if len(good_results) >= COVERAGE_THRESHOLD else "insufficient",
        "good_count": len(good_results),
        "best_score": results[0]["score"]
    }


def decide_action(evaluation: dict, step: int, max_steps: int) -> str:
    if step >= max_steps - 1:
        return "stop"
    if evaluation["status"] == "sufficient":
        return "answer"
    if evaluation["status"] == "empty":
        return "broaden"
    return "refine"


def refine_query(query: str, attempt: int) -> str:
    strategies = [
        lambda q: q + " explanation and examples",
        lambda q: "detailed overview of " + q,
        lambda q: "how does " + q + " work in practice",
    ]
    return strategies[attempt % len(strategies)](query)


def broaden_query(query: str) -> str:
    words = query.split()
    return " ".join(words[:3]) if len(words) > 3 else query + " overview"


def generate_answer(query: str, results: list[dict]) -> str:
    context_lines = [f"[{i+1}] {r['text']} (source: {r['source']})"
                     for i, r in enumerate(results)]
    context = "\n".join(context_lines)
    return (
        f"Based on the retrieved context, here is the answer to: '{query}'\n\n"
        f"Context used:\n{context}\n\n"
        f"[In production: this context + question would be sent to an LLM for final generation]"
    )


def agentic_rag(query: str, max_steps: int = 4) -> str:
    """
    Agentic RAG decision loop.
    Retrieves, evaluates, and decides whether to answer, refine, broaden, or stop.
    """
    print(f"{'='*60}")
    print(f"  Agentic RAG — Query: '{query}'")
    print(f"{'='*60}\n")

    current_query = query
    all_results = []

    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        print(f"Query: '{current_query}'")

        results = retrieve(current_query, knowledge_base, top_k=3)
        print(f"Retrieved {len(results)} documents:")
        for r in results:
            print(f"  [{r['score']}] {r['text'][:70]}...")

        # Accumulate unique results across steps
        existing_sources = {r["source"] for r in all_results}
        new_results = [r for r in results if r["source"] not in existing_sources]
        all_results.extend(new_results)
        print(f"Total unique results so far: {len(all_results)}")

        evaluation = evaluate_results(results)
        print(f"Evaluation: {evaluation}")

        action = decide_action(evaluation, step, max_steps)
        print(f"Agent decision: {action.upper()}\n")

        if action == "answer":
            print("Sufficient context found. Generating answer...\n")
            return generate_answer(query, all_results[:5])
        elif action == "refine":
            current_query = refine_query(current_query, step)
            print(f"Refining query to: '{current_query}'\n")
        elif action == "broaden":
            current_query = broaden_query(current_query)
            print(f"Broadening query to: '{current_query}'\n")
        elif action == "stop":
            break

    if all_results:
        print("Max steps reached. Using best available results...\n")
        return generate_answer(query, all_results[:5])

    return "Agent could not find sufficient information to answer this query."


if __name__ == "__main__":
    # Test 1: Query that needs refinement
    answer = agentic_rag("how does RAG work?", max_steps=4)
    print("\nFINAL ANSWER:")
    print(answer)

    print("\n" + "=" * 60 + "\n")

    # Test 2: Query that resolves immediately
    answer2 = agentic_rag("search systems", max_steps=4)
    print("\nFINAL ANSWER:")
    print(answer2)
```

---

## Expected Output

```
============================================================
  Agentic RAG — Query: 'how does RAG work?'
============================================================

--- Step 1 ---
Query: 'how does RAG work?'
Retrieved 3 documents:
  [0.4677] Agentic RAG uses a decision loop to dynamically decide when and what t...
  [0.4414] RAG improves LLM accuracy by retrieving relevant documents before gene...
  [0.1297] Context construction organizes retrieved chunks into a clean, token-aw...
Total unique results so far: 3
Evaluation: {'status': 'insufficient', 'good_count': 1, 'best_score': 0.4677}
Agent decision: REFINE

Refining query to: 'how does RAG work? explanation and examples'

--- Step 2 ---
...
Agent decision: REFINE

--- Step 3 ---
...
Evaluation: {'status': 'sufficient', 'good_count': 2, 'best_score': 0.5}
Agent decision: ANSWER

Sufficient context found. Generating answer...
```

The agent refined the query twice before finding sufficient context. The second test query resolved immediately in step 1 — the agent recognized good results and answered without iterating.

---

## What Each Part Does

**`evaluate_results(results)`**
Judges whether retrieved results are good enough. Counts how many results are above the relevance threshold (`0.45`) and whether there's enough coverage (`>= 2` good results). Returns `sufficient`, `insufficient`, or `empty`.

**`decide_action(evaluation, step, max_steps)`**
The agent's brain. Maps evaluation status to an action. Enforces the step limit — if max steps are reached, it stops regardless of result quality.

**`refine_query(query, attempt)`**
Adds specificity to a weak query. Three strategies rotate across attempts: adding "explanation and examples", prepending "detailed overview of", and appending "work in practice". In production, an LLM would generate smarter refinements.

**`broaden_query(query)`**
Simplifies an overly specific query that returned nothing. Keeps only the first 3 words or appends "overview".

**`agentic_rag(query, max_steps)`**
The main loop. Accumulates unique results across iterations — so even if step 1 returns weak results, those chunks are kept and combined with better results from later steps.

---

## Standard RAG vs Agentic RAG

| | Standard RAG | Agentic RAG |
|---|---|---|
| Flow | Linear, fixed | Dynamic, loop-based |
| Retrieval attempts | Once | Up to max_steps |
| Query adaptation | None | Refine or broaden |
| Handles bad retrieval | No | Yes — retries |
| Handles ambiguous queries | Poorly | Better |
| Complexity | Low | Medium |
| Best for | Simple, clear queries | Complex, multi-part queries |

---

## Taking It Further in Production

This implementation uses rule-based query refinement. In a real production system:

**LLM-powered decisions**
Replace `decide_action` and `refine_query` with LLM calls. The LLM reads the retrieved results and decides what to do next — much more intelligent than threshold-based rules.

**Tool use**
A production agent might have multiple tools: web search, database lookup, calculator, code execution. It picks the right tool for each step.

**Memory across turns**
Store what was retrieved in previous steps so the agent doesn't re-retrieve the same documents.

**Confidence scoring**
Instead of a binary threshold, use a confidence score that accounts for result diversity, source quality, and query-result alignment.

---

## Key Takeaways

- Standard RAG is linear — it retrieves once and generates. Agentic RAG loops until satisfied
- The agent evaluates retrieved results and decides: answer, refine, broaden, or stop
- Query refinement adds specificity; query broadening removes constraints
- Results accumulate across iterations — each step builds on the previous
- A step limit prevents infinite loops — the agent always terminates
- Rule-based decisions work for demos; LLM-powered decisions work better in production
- Agentic RAG handles complex, ambiguous, and multi-part queries that fixed pipelines can't

---

## What's Next

On **Day 20**, we go even deeper — Graph RAG, where knowledge is stored not as flat document chunks but as a connected graph of entities and relationships. This enables retrieval that follows connections, not just similarity.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
