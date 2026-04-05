"""
Day 20 — Graph RAG

Demonstrates how a knowledge graph enables retrieval of connected information
that flat vector search would miss.

No heavy frameworks — pure Python using dicts and lists.
"""

# ---------------------------------------------------------------------------
# Knowledge Graph
# Nodes = entities/concepts
# Edges = relationships between them (directed)
# ---------------------------------------------------------------------------

graph = {
    # RAG system components
    "rag system": {
        "description": "Retrieval Augmented Generation combines retrieval and generation for accurate answers.",
        "related_to": ["retriever", "llm", "knowledge base", "context construction"]
    },
    "retriever": {
        "description": "Searches the knowledge base and returns the most relevant document chunks.",
        "related_to": ["embedding model", "vector database", "similarity search", "reranking"]
    },
    "llm": {
        "description": "Large Language Model that generates answers from retrieved context.",
        "related_to": ["prompt", "context construction", "hallucination"]
    },
    "knowledge base": {
        "description": "The collection of documents, chunks, and vectors that the retriever searches.",
        "related_to": ["document ingestion", "chunking", "vector database"]
    },
    "embedding model": {
        "description": "Converts text into numerical vectors that capture semantic meaning.",
        "related_to": ["vector database", "similarity search"]
    },
    "vector database": {
        "description": "Stores embedding vectors and enables fast nearest-neighbor search.",
        "related_to": ["similarity search", "metadata filtering"]
    },
    "similarity search": {
        "description": "Finds the most semantically similar documents by comparing vectors.",
        "related_to": ["cosine similarity", "embedding model"]
    },
    "cosine similarity": {
        "description": "Measures the angle between two vectors to determine semantic closeness.",
        "related_to": ["similarity search"]
    },
    "reranking": {
        "description": "Re-scores retrieved documents for precision before passing to the LLM.",
        "related_to": ["retriever", "context construction"]
    },
    "context construction": {
        "description": "Organizes retrieved chunks into a clean, token-aware block for the LLM.",
        "related_to": ["llm", "prompt", "reranking"]
    },
    "prompt": {
        "description": "Instructions that tell the LLM how to use retrieved context to answer.",
        "related_to": ["llm", "hallucination"]
    },
    "hallucination": {
        "description": "When an LLM generates confident but factually incorrect information.",
        "related_to": ["llm", "rag system"]
    },
    "chunking": {
        "description": "Splitting large documents into smaller pieces for more precise retrieval.",
        "related_to": ["knowledge base", "document ingestion"]
    },
    "document ingestion": {
        "description": "Pipeline that loads, cleans, and prepares raw documents for indexing.",
        "related_to": ["chunking", "knowledge base"]
    },
    "metadata filtering": {
        "description": "Narrows the search space using structured attributes like date or source.",
        "related_to": ["vector database", "retriever"]
    },
    "hybrid search": {
        "description": "Combines keyword search and semantic search for better recall.",
        "related_to": ["retriever", "similarity search"]
    },
}


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def find_node(query: str) -> str | None:
    """
    Find the best matching node for a query.
    Checks for exact match first, then partial match.
    """
    query_lower = query.lower().strip()

    # Exact match
    if query_lower in graph:
        return query_lower

    # Partial match — find nodes that contain the query words
    query_words = set(query_lower.split())
    best_match = None
    best_overlap = 0

    for node in graph:
        node_words = set(node.split())
        overlap = len(query_words & node_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = node

    return best_match if best_overlap > 0 else None


def traverse_graph(start_node: str, depth: int = 2) -> list[dict]:
    """
    BFS traversal from a starting node up to a given depth.
    Returns all visited nodes with their descriptions and depth level.
    """
    visited = set()
    results = []
    queue = [(start_node, 0)]  # (node, current_depth)

    while queue:
        node, current_depth = queue.pop(0)

        if node in visited or current_depth > depth:
            continue

        visited.add(node)

        if node in graph:
            results.append({
                "node": node,
                "description": graph[node]["description"],
                "depth": current_depth,
                "connections": graph[node]["related_to"]
            })

            # Add neighbors to queue for next depth level
            if current_depth < depth:
                for neighbor in graph[node]["related_to"]:
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))

    return results


def build_graph_context(traversal_results: list[dict]) -> str:
    """
    Format traversal results into a readable context block for the LLM.
    Groups by depth level so the LLM understands the relationship hierarchy.
    """
    if not traversal_results:
        return "No relevant information found in the knowledge graph."

    lines = []
    current_depth = -1

    for result in traversal_results:
        if result["depth"] != current_depth:
            current_depth = result["depth"]
            label = "Direct match" if current_depth == 0 else f"Connected (depth {current_depth})"
            lines.append(f"\n[{label}]")

        lines.append(f"  • {result['node'].title()}: {result['description']}")
        if result["connections"]:
            lines.append(f"    → connects to: {', '.join(result['connections'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph RAG pipeline
# ---------------------------------------------------------------------------

def graph_rag(query: str, depth: int = 2) -> dict:
    """
    Full Graph RAG pipeline:
    Query → Find matching node → Traverse graph → Build context → Ready for LLM
    """
    print(f"Query: '{query}'")
    print("-" * 50)

    # Step 1: Find the entry node
    start_node = find_node(query)
    if not start_node:
        print("No matching node found in knowledge graph.\n")
        return {"query": query, "context": "No relevant information found.", "nodes_visited": 0}

    print(f"Entry node: '{start_node}'")

    # Step 2: Traverse the graph
    traversal = traverse_graph(start_node, depth=depth)
    print(f"Nodes visited: {len(traversal)} (depth={depth})")
    for r in traversal:
        indent = "  " * r["depth"]
        print(f"{indent}[depth {r['depth']}] {r['node']}")

    # Step 3: Build context
    context = build_graph_context(traversal)

    return {
        "query": query,
        "entry_node": start_node,
        "nodes_visited": len(traversal),
        "context": context
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Graph RAG Demo")
    print("=" * 60 + "\n")

    # Test 1: Query about the RAG system
    result = graph_rag("rag system", depth=2)
    print("\nGraph Context:")
    print(result["context"])

    print("\n" + "=" * 60 + "\n")

    # Test 2: Query about hallucination — traverses back to RAG system
    result2 = graph_rag("hallucination", depth=2)
    print("\nGraph Context:")
    print(result2["context"])

    print("\n" + "=" * 60 + "\n")

    # Test 3: Show what flat vector search would miss
    print("What flat vector search misses:")
    print("  Query: 'hallucination'")
    print("  Flat search returns: chunks about hallucination only")
    print("  Graph RAG returns: hallucination + LLM + prompt + RAG system")
    print("  The connections reveal WHY hallucination happens and HOW to prevent it.")
