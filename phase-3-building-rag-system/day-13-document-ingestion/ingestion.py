from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({
                "text": f.read(),
                "source": file.name
            })
    return docs


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_documents(folder: str) -> list[dict]:
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
    if data:
        print("\nSample chunk:")
        print(f"  Text: {data[0]['text'][:100]}...")
        print(f"  Metadata: {data[0]['metadata']}")
