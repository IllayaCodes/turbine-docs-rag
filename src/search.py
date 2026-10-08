import sys
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
TOP_K = 5
SNIPPET_CHARS = 700


def load_index(strategy):
    chunks = json.loads((CHUNKS_DIR / f"{strategy}.json").read_text(encoding="utf-8"))
    embeddings = np.load(INDEX_DIR / f"{strategy}_embeddings.npy")
    meta = json.loads((INDEX_DIR / f"{strategy}_meta.json").read_text(encoding="utf-8"))
    return chunks, embeddings, meta


def search(query, model, chunks, embeddings, top_k=TOP_K):
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vector
    best = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in best]


def show_results(results):
    for rank, (score, chunk) in enumerate(results, start=1):
        snippet = chunk["text"][:SNIPPET_CHARS].replace("\n", " ")
        print(f"\n#{rank}  score {score:.3f}  [{chunk['source']}, page {chunk['page']}]  ({chunk['chunk_id']})")
        print(f"    {snippet}...")


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sentences"
    try:
        chunks, embeddings, meta = load_index(strategy)
    except FileNotFoundError as error:
        print(f"Missing file: {error.filename}. Run build_index.py {strategy} first.")
        return

    print(f"Loading model {meta['model']}...")
    model = SentenceTransformer(meta["model"])
    print(f"Ready: {meta['chunks']} chunks searchable ({strategy}). Press Enter on an empty line to quit.")

    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            break
        show_results(search(query, model, chunks, embeddings))


if __name__ == "__main__":
    main()