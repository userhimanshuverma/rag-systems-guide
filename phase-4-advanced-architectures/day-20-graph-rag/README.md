# Day 20 — Graph RAG

> "Flat search finds similar things. Graph search finds connected things. Those are very different."

---

## The Problem

Standard RAG retrieves document chunks based on semantic similarity. You ask about "hallucination" and it returns chunks about hallucination. That's useful — but it's isolated.

What it misses is the *why* and the *how*.

Why do hallucinations happen? Because the LLM generates from memory when context is missing. How do you prevent them? With better prompts, better retrieval, and grounding the LLM in real context.

That full picture — the connections between concepts — lives in the relationships between entities, not in any single document chunk. Flat vector search can't see it. It only finds what's similar, not what's connected.

Graph RAG solves this by storing knowledge as a network of entities and relationships, then traversing that network to retrieve connected information.

---

## What Graph RAG Actually Is

In standard RAG, knowledge is stored as flat chunks:

```
Chunk 1: "Hallucinations occur when LLMs generate incorrect information."
Chunk 2: "RAG reduces hallucinations by grounding answers in retrieved context."
```

In Graph RAG, knowledge is stored as a connected graph:

```
hallucination ──→ llm ──→ prompt
      └──────────→ rag system ──→ retriever ──→ knowledge base
```

When you query "hallucination", you don't just get the hallucination chunk. You traverse the graph and retrieve everything connected to it — the LLM, the prompt, the RAG system, the retriever. The full picture.

> Graph RAG = retrieve the node + traverse its connections + aggregate the context

---

## Nodes and Edges

**Nodes** are entities — concepts, components, people, events, anything that can be named.

```
"hallucination", "llm", "rag system", "retriever", "prompt"
```

**Edges** are relationships between nodes — directed connections that say "this relates to that."

```
hallucination → llm          (hallucination is caused by the LLM)
hallucination → rag system   (RAG system addresses hallucination)
llm → prompt                 (LLM behavior is controlled by prompts)
```

The graph captures not just what things are, but how they relate to each other. That's the information flat search can't retrieve.

---

## Architecture Flow

```
User Query
    ↓
Find Entry Node
(match query to a node in the graph)
    ↓
Graph Traversal (BFS/DFS)
(follow edges to connected nodes, up to depth N)
    ↓
Collect Connected Knowledge
(all visited nodes + their descriptions)
    ↓
Build Context Block
(format for LLM, grouped by depth)
    ↓
LLM → Final Answer
(with full connected context)
```

---

## The Code

No external dependencies — pure Python.

```python
# Knowledge Graph
# Nodes = entities, Edges = relationships
graph = {
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
    "hallucination": {
        "description": "When an LLM generates confident but factually incorrect information.",
        "related_to": ["llm", "rag system"]
    },
    "prompt": {
        "description": "Instructions that tell the LLM how to use retrieved context to answer.",
        "related_to": ["llm", "hallucination"]
    },
    "embedding model": {
        "description": "Converts text into numerical vectors that capture semantic meaning.",
        "related_to": ["vector database", "similarity search"]
    },
    "vector database": {
        "description": "Stores embedding vectors and enables fast nearest-neighbor search.",
        "related_to": ["similarity search", "metadata filtering"]
    },
    # ... more nodes
}


def find_node(query: str) -> str | None:
    """Find the best matching node for a query — exact match first, then partial."""
    query_lower = query.lower().strip()
    if query_lower in graph:
        return query_lower

    query_words = set(query_lower.split())
    best_match, best_overlap = None, 0
    for node in graph:
        overlap = len(query_words & set(node.split()))
        if overlap > best_overlap:
            best_overlap, best_match = overlap, node
    return best_match if best_overlap > 0 else None


def traverse_graph(start_node: str, depth: int = 2) -> list[dict]:
    """
    BFS traversal from a starting node up to a given depth.
    Returns all visited nodes with descriptions and depth level.
    """
    visited = set()
    results = []
    queue = [(start_node, 0)]

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
            if current_depth < depth:
                for neighbor in graph[node]["related_to"]:
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
    return results


def build_graph_context(traversal_results: list[dict]) -> str:
    """Format traversal results into a readable context block for the LLM."""
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


def graph_rag(query: str, depth: int = 2) -> dict:
    """Full Graph RAG pipeline: Query → Node → Traverse → Context"""
    print(f"Query: '{query}'")
    print("-" * 50)

    start_node = find_node(query)
    if not start_node:
        return {"query": query, "context": "No relevant information found.", "nodes_visited": 0}

    print(f"Entry node: '{start_node}'")
    traversal = traverse_graph(start_node, depth=depth)
    print(f"Nodes visited: {len(traversal)} (depth={depth})")

    context = build_graph_context(traversal)
    return {"query": query, "entry_node": start_node, "nodes_visited": len(traversal), "context": context}


if __name__ == "__main__":
    # Query about hallucination — traverses to LLM, prompt, RAG system
    result = graph_rag("hallucination", depth=2)
    print("\nGraph Context:")
    print(result["context"])
```

---

## Expected Output

```
Query: 'hallucination'
--------------------------------------------------
Entry node: 'hallucination'
Nodes visited: 7 (depth=2)

Graph Context:

[Direct match]
  • Hallucination: When an LLM generates confident but factually incorrect information.
    → connects to: llm, rag system

[Connected (depth 1)]
  • Llm: Large Language Model that generates answers from retrieved context.
    → connects to: prompt, context construction, hallucination
  • Rag System: Retrieval Augmented Generation combines retrieval and generation for accurate answers.
    → connects to: retriever, llm, knowledge base, context construction

[Connected (depth 2)]
  • Prompt: Instructions that tell the LLM how to use retrieved context to answer.
    → connects to: llm, hallucination
  • Context Construction: Organizes retrieved chunks into a clean, token-aware block for the LLM.
  • Retriever: Searches the knowledge base and returns the most relevant document chunks.
  • Knowledge Base: The collection of documents, chunks, and vectors that the retriever searches.
```

A flat vector search for "hallucination" returns one chunk. Graph RAG returns 7 connected nodes — the full picture of what causes hallucinations and how the RAG system addresses them.

---

## What Each Part Does

**`graph` (the knowledge graph)**
A dictionary where each key is a node (entity) and each value contains a description and a list of related nodes (edges). This is the simplest possible graph representation — no special library needed.

**`find_node(query)`**
Maps a user query to a graph node. Tries exact match first, then partial word overlap. In production, you'd use semantic search here — embed the query and find the closest node embedding.

**`traverse_graph(start_node, depth)`**
BFS (Breadth-First Search) traversal starting from the entry node. Visits all connected nodes up to the specified depth. Depth 1 = direct neighbors. Depth 2 = neighbors of neighbors. Returns every visited node with its description and depth level.

**`build_graph_context(traversal_results)`**
Formats the traversal results into a structured context block grouped by depth. The LLM sees the direct match first, then depth-1 connections, then depth-2 connections — a natural hierarchy of relevance.

**`graph_rag(query, depth)`**
The full pipeline. Find node → traverse → build context → return for LLM.

---

## Vector RAG vs Graph RAG

| | Vector RAG | Graph RAG |
|---|---|---|
| Knowledge structure | Flat chunks | Connected nodes + edges |
| Retrieval method | Similarity search | Node matching + traversal |
| What it finds | Similar content | Connected content |
| Best for | Factual Q&A, document search | Relationship queries, root cause analysis |
| Handles "why" questions | Poorly | Well |
| Handles "how does X relate to Y" | Poorly | Well |
| Setup complexity | Low | Medium |
| Scales to large corpora | Easily | Requires careful graph design |

They're not competitors — they're complements. Many production systems use both: vector search for broad retrieval, graph traversal for relationship-aware context enrichment.

---

## Real-World Use Cases

**Root cause analysis**
An incident occurs. Query "database timeout" and traverse to: connection pool → query optimizer → slow queries → missing indexes. The graph reveals the chain of causation that flat search can't.

**Knowledge management systems**
Query "onboarding process" and traverse to: HR policies → IT setup → team introductions → tools access. The full connected picture, not just one document.

**Medical knowledge**
Query "chest pain" and traverse to: symptoms → conditions → treatments → contraindications. Relationships between medical concepts that isolated chunks can't capture.

**Compliance and legal**
Query "GDPR Article 17" and traverse to: related articles → case law → implementation guidelines → affected systems. The regulatory web, not just one clause.

---

## Trade-offs

**Graph RAG is more powerful for relationship queries** — but it requires upfront work to build and maintain the graph. Nodes and edges don't appear automatically. Someone (or an LLM pipeline) has to extract them from documents.

**Depth matters** — depth 1 gives direct connections, depth 2 gives the broader picture, depth 3+ can get noisy. Tuning depth for your use case is important.

**Graph quality determines answer quality** — a poorly constructed graph with wrong or missing edges produces misleading context. The graph is your knowledge model — it needs to be accurate.

---

## Key Takeaways

- Graph RAG stores knowledge as nodes (entities) and edges (relationships), not flat chunks
- Traversal retrieves connected knowledge — not just similar content, but related content
- A query about "hallucination" surfaces LLM, prompt, RAG system — the full causal picture
- BFS traversal with depth control lets you tune how broadly to explore the graph
- Vector RAG and Graph RAG are complementary — many systems use both
- Graph RAG excels at "why", "how does X relate to Y", and root cause analysis queries
- Building the graph is the hard part — but the retrieval quality payoff is significant
- RAG is not just retrieval. It's understanding connections.

---

## What's Next

Tomorrow is the final day of the series.

On **Day 21**, we bring everything together — Production RAG Systems. Architecture for scaling, caching, monitoring, evaluation, latency optimization, and cost management. Everything you need to take a RAG system from prototype to production.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
