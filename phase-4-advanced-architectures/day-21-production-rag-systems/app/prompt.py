"""
Prompt templates for the RAG system.
Provides strict grounding instructions to minimize hallucination.
"""


def build_rag_prompt(context: str, question: str) -> str:
    """
    Standard RAG prompt.
    Instructs the LLM to answer only from provided context.
    """
    return f"""You are a precise and helpful assistant.

INSTRUCTIONS:
- Answer the question using ONLY the context provided below.
- If the answer is not present in the context, respond with: "I don't know based on the provided context."
- Do not use any outside knowledge or make assumptions.
- Be concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_summary_prompt(context: str, topic: str) -> str:
    """Summarization prompt — synthesizes context into a concise summary."""
    return f"""You are a helpful assistant that summarizes technical content.

Using ONLY the context below, write a concise summary about: {topic}
Keep it under 150 words. Do not add information not present in the context.

CONTEXT:
{context}

SUMMARY:"""
