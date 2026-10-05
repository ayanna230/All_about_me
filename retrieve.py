"""
Step 3: retrieval.

Given a user's question:
1. Embed the question with the SAME model used for the chunks
   (mismatched models would produce vectors that aren't comparable).
2. Compare the question's vector against every chunk vector using a dot
   product (works as cosine similarity because everything is normalized).
3. Return the top-k highest scoring chunks.

No vector database needed at this scale -- 41 chunks fits trivially in
memory, and a dot product against a NumPy array is already extremely fast.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Windows consoles often default to a legacy codepage (cp1252) that can't
# represent characters like em dashes or arrows used in the source doc.
# Force stdout to UTF-8 so printing chunk text never crashes.
sys.stdout.reconfigure(encoding="utf-8")

EMBEDDINGS_FILE = Path("embeddings.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_index():
    data = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    vectors = np.array([c["embedding"] for c in data])
    return data, vectors


def retrieve(query: str, model, chunks, vectors, top_k: int = 3):
    query_vector = model.encode([query], normalize_embeddings=True)[0]

    # dot product of a normalized query against every normalized chunk vector
    # = cosine similarity for every chunk, all at once (no loop needed)
    scores = vectors @ query_vector

    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for i in top_indices:
        results.append({
            "score": float(scores[i]),
            "breadcrumb": chunks[i]["breadcrumb"],
            "text": chunks[i]["text"],
        })
    return results


def main():
    chunks, vectors = load_index()
    print(f"Loaded {len(chunks)} chunks into the retrieval index.")

    model = SentenceTransformer(MODEL_NAME)

    print("\nType a question about Ayanna (or 'quit' to exit).")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break

        results = retrieve(query, model, chunks, vectors, top_k=3)
        for r in results:
            print(f"\n[{r['score']:.3f}] {r['breadcrumb']}")
            print(r["text"])


if __name__ == "__main__":
    main()
