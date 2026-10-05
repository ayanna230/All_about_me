"""
Step 2: turn each chunk's text into an embedding vector.

Model: all-MiniLM-L6-v2 (via sentence-transformers)
- Runs locally, no API key.
- Produces a 384-dimensional vector per chunk.
- Good balance of speed/quality for a small personal-doc project.

We save the vectors alongside the chunk metadata in embeddings.json so the
next step (retrieval) can load everything without re-embedding.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("chunks.json")
OUTPUT_FILE = Path("embeddings.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(f"Loading embedding model '{MODEL_NAME}' (downloads once, then cached)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]

    print("Embedding all chunks...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()

    OUTPUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(chunks)} chunks with embeddings to {OUTPUT_FILE}")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")


if __name__ == "__main__":
    main()
