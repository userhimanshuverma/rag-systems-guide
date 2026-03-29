# Day 13 — Document Ingestion Pipelines

> "Before you can search your data, you have to prepare it. Raw data is just noise."

---

## The Problem

You have documents. PDFs, text files, support tickets, internal wikis, database exports. You want your RAG system to answer questions from all of this.

But you can't just point your system at a folder and call it done.

Raw data is messy. PDFs have headers, footers, and page numbers mixed into the content. Text files have inconsistent formatting. Some documents are 50 pages long — way too large to embed as a single unit. Others have no context about where they came from.

Before any of this data can be retrieved, it needs to be prepared. Cleaned. Split into the right sizes. Tagged with metadata. Structured in a way the retrieval system can actually use.

That preparation process is the ingestion pipeline.

---

## What Ingestion Actually Is

Ingestion is the process of taking raw, unstructured data and transforming it into clean, structured chunks that are ready to be embedded and stored.

Think of it as the data preparation layer — everything that happens before indexing.

> Raw data → Ingestion Pipeline → Structured chunks ready for retrieval

Without a solid ingestion pipeline, even the best retrieval system will return garbage. The quality of your RAG system starts here.

---

## Types of Data Sources

Real-world RAG systems ingest from many different sources:

**Text files** — the simplest case. Plain `.txt` files with clean content.

**PDFs** — common but tricky. Need to extract text while handling layout, columns, headers, and footers.

**Word documents / Markdown** — structured documents with headings and sections that can guide chunking.

**Web pages / HTML** — need to strip navigation, ads, and boilerplate to get to the actual content.

**Databases** — structured records that need to be converted into readable text chunks.

**APIs** — live data sources like support tickets, CRM records, or product catalogs.

**Logs** — semi-structured text that needs parsing before it's useful.

Each source type needs its own loader — a piece of code that knows how to extract clean text from that format.

---

## The Cooking Analogy

Think of ingestion like preparing ingredients before cooking.

You don't throw a whole raw chicken, an unpeeled onion, and a bag of rice into a pot and call it a meal. You clean, chop, measure, and prep everything first. Only then does cooking (retrieval + generation) produce something good.

Your documents are the raw ingredients. The ingestion pipeline is the prep work. The vector database is the organized pantry. And the LLM is the chef.

Skip the prep, and the meal is a mess.

---

## The Ingestion Pipeline Step by Step

**Step 1 — Load**
Read the raw file and extract its text content. Different loaders handle different file types.

**Step 2 — Clean**
Remove noise — extra whitespace, page numbers, headers, footers, HTML tags, special characters. Keep only the meaningful content.

**Step 3 — Chunk**
Split the cleaned text into smaller pieces. Each chunk should be small enough to embed meaningfully but large enough to contain useful context. Overlap between chunks helps preserve context at boundaries.

**Step 4 — Add Metadata**
Tag each chunk with information about where it came from — source file, date, category, page number, etc. This metadata enables filtering later.

**Step 5 — Ready for Indexing**
The processed chunks are now structured objects — text + metadata — ready to be embedded and stored in the vector database.

---

## Architecture Flow

```
Raw Data (PDFs, TXT, APIs, DBs)
        ↓
    Loader
(extract text from source format)
        ↓
  Preprocessing
(clean noise, normalize text)
        ↓
    Chunking
(split into right-sized pieces with overlap)
        ↓
Metadata Enrichment
(tag with source, date, category, etc.)
        ↓
Structured Chunks
(text + metadata, ready for indexing)
        ↓
  → Indexing Pipeline (Day 14)
```

---

## The Code

Here's a clean, minimal implementation of a document ingestion pipeline — no heavy frameworks, just Python.

```python
from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    """
    Load all .txt files from a folder.
    Returns a list of dicts with text content and source filename.
    """
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({
                "text": f.read(),
                "source": file.name
            })
    return docs


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Removes extra whitespace and normalizes line breaks.
    """
    # Collapse multiple newlines into one
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    chunk_size: number of characters per chunk
    overlap: how many characters to repeat between consecutive chunks
             (preserves context at chunk boundaries)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but keep some overlap

    return chunks


def process_documents(folder: str) -> list[dict]:
    """
    Full ingestion pipeline:
    Load → Clean → Chunk → Attach Metadata
    """
    raw_docs = load_documents(folder)
    processed = []

    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i
                }
            })

    return processed


if __name__ == "__main__":
    data = process_documents("./data")
    print(f"Total chunks ready for indexing: {len(data)}")

    # Preview first chunk
    if data:
        print("\nSample chunk:")
        print(f"  Text: {data[0]['text'][:100]}...")
        print(f"  Metadata: {data[0]['metadata']}")
```

---

## What Each Function Does

**`load_documents(folder)`**
Reads every `.txt` file in a folder and returns the raw text along with the filename. The filename becomes part of the metadata — so you always know where a chunk came from.

**`clean_text(text)`**
Strips out empty lines and extra whitespace. In real systems this gets more sophisticated — removing HTML tags, page numbers, headers — but the principle is the same: keep only meaningful content.

**`chunk_text(text, chunk_size, overlap)`**
Splits text into fixed-size chunks with overlap. The overlap is important — if a sentence spans a chunk boundary, the overlap ensures neither chunk loses that context entirely. `chunk_size=300` and `overlap=50` are reasonable starting defaults for character-based chunking.

**`process_documents(folder)`**
The main pipeline function. Calls the others in sequence: load → clean → chunk → attach metadata. Returns a list of structured chunk objects ready for the next stage.

---

## Why Chunking + Metadata Matter

**Chunking** controls what the embedding model sees. Too large and the embedding captures too many topics at once — retrieval becomes imprecise. Too small and chunks lose context — retrieved chunks don't make sense on their own. The right chunk size depends on your content and use case.

**Overlap** prevents information loss at boundaries. If a key sentence falls right at the edge of a chunk, overlap ensures it appears in both the current and next chunk — so retrieval doesn't miss it.

**Metadata** is what enables filtering later. Without `source` attached to every chunk, you can't filter by document. Without `chunk_index`, you can't reconstruct the original order. Metadata is cheap to add during ingestion and invaluable during retrieval.

---

## Key Takeaways

- Ingestion is the process of transforming raw data into structured chunks ready for retrieval
- The pipeline has five stages: Load → Clean → Chunk → Add Metadata → Ready for Indexing
- Different data sources need different loaders — PDFs, text files, APIs, databases all require different extraction logic
- Chunk size and overlap are tunable parameters — they directly affect retrieval quality
- Overlap between chunks preserves context at boundaries and prevents information loss
- Metadata attached during ingestion enables filtering, tracing, and debugging later
- Clean ingestion = clean retrieval = clean answers — quality starts here
- You don't need a heavy framework to build this — a few clean Python functions are enough to start

---

## What's Next

Your documents are now clean, chunked, and tagged with metadata.

On **Day 14**, we'll cover the Indexing Pipeline — taking those processed chunks, converting them into embeddings, and storing them in a vector database so they can actually be searched.

See you there.

---

*Part of the [21-Day RAG Systems Series](../../README.md)*
