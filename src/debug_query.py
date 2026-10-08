import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from search import load_index

sys.stdout.reconfigure(encoding="utf-8")

STRATEGY = "sentences"
SHOW_TOP = 5


def main():
    if len(sys.argv) != 4:
        print('Usage: python src\\debug_query.py "your question" DOCUMENT_ID PAGE')
        print('Example: python src\\debug_query.py "What is the pinch point?" 20100029602 11')
        return

    query, document_id, page = sys.argv[1], sys.argv[2], int(sys.argv[3])
    chunks, embeddings, meta = load_index(STRATEGY)
    model = SentenceTransformer(meta["model"])

    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vector
    order = np.argsort(scores)[::-1]
    rank_of = {int(index): rank for rank, index in enumerate(order, start=1)}

    print(f"\nQuestion: {query}")
    print(f"\nTop {SHOW_TOP} overall:")
    for index in order[:SHOW_TOP]:
        c = chunks[int(index)]
        print(f"  #{rank_of[int(index)]}  score {scores[index]:.3f}  {c['chunk_id']}")

    print(f"\nChunks from {document_id}, page {page} (out of {len(chunks)} total):")
    found = False
    for index, c in enumerate(chunks):
        if c["source"].startswith(document_id) and c["page"] == page:
            found = True
            print(f"\n  rank #{rank_of[index]}  score {scores[index]:.3f}  {c['chunk_id']}")
            print(f"  {c['text']}")
    if not found:
        print("  (no chunks found for that document and page)")


if __name__ == "__main__":
    main()