"""
Day 17 — Prompting with Retrieved Knowledge

Demonstrates different prompt patterns for RAG systems:
- Basic prompt
- Strict grounding prompt (no hallucination)
- Multi-source citation prompt
- Conversational prompt with history
"""


# ---------------------------------------------------------------------------
# Sample context (output from Day 16 context construction)
# ---------------------------------------------------------------------------
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
    """
    Basic RAG prompt.
    Instructs the LLM to use context and admit when it doesn't know.
    """
    return f"""You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Pattern 2 — Strict Grounding Prompt
# Maximally reduces hallucination — forces citation
# ---------------------------------------------------------------------------

def build_strict_prompt(context: str, question: str) -> str:
    """
    Strict grounding prompt.
    Forces the LLM to cite the source number for every claim it makes.
    Best for high-stakes use cases (legal, medical, compliance).
    """
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
# When you want a concise synthesis, not a direct answer
# ---------------------------------------------------------------------------

def build_summary_prompt(context: str, topic: str) -> str:
    """
    Summarization prompt.
    Asks the LLM to synthesize the context into a concise summary on a topic.
    """
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
    Conversational RAG prompt with chat history.
    Includes previous turns so the LLM can maintain context across messages.

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
# Demonstrate all patterns
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


if __name__ == "__main__":
    question = "How does RAG improve LLM answers?"

    # Pattern 1 — Basic
    separator("Pattern 1: Basic RAG Prompt")
    print(build_basic_prompt(sample_context, question))

    # Pattern 2 — Strict with citations
    separator("Pattern 2: Strict Grounding Prompt")
    print(build_strict_prompt(sample_context, question))

    # Pattern 3 — Summarization
    separator("Pattern 3: Summarization Prompt")
    print(build_summary_prompt(sample_context, "how RAG systems work"))

    # Pattern 4 — Conversational with history
    separator("Pattern 4: Conversational Prompt")
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating an answer."},
    ]
    print(build_conversational_prompt(sample_context, "And how does it reduce hallucinations?", history))