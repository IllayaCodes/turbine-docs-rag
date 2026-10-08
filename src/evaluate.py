import sys
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search import load_index

sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"
CUTOFFS = (1, 3, 5, 10)
MRR_CUTOFF = 10


def normalize(text):
    return " ".join(text.lower().split())


def is_relevant(chunk, question):
    if not chunk["source"].startswith(question["document"]):
        return False
    text = normalize(chunk["text"])
    return any(normalize(phrase) in text for phrase in question["answer_phrases"])


def rank_of_first_relevant(order, chunks, question):
    for rank, index in enumerate(order, start=1):
        if is_relevant(chunks[int(index)], question):
            return rank
    return None


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sentences"
    label = sys.argv[2] if len(sys.argv) > 2 else strategy

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    chunks, embeddings, meta = load_index(strategy)
    model = SentenceTransformer(meta["model"])

    vectors = model.encode([q["question"] for q in questions], normalize_embeddings=True)

    rows = []
    for question, vector in zip(questions, vectors):
        scores = embeddings @ vector
        order = np.argsort(scores)[::-1]
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "rank": rank_of_first_relevant(order, chunks, question),
            "top_chunk": chunks[int(order[0])]["chunk_id"],
        })

    total = len(rows)
    metrics = {}
    for k in CUTOFFS:
        metrics[f"hit@{k}"] = sum(1 for r in rows if r["rank"] and r["rank"] <= k) / total
    metrics[f"mrr@{MRR_CUTOFF}"] = sum(
        1 / r["rank"] for r in rows if r["rank"] and r["rank"] <= MRR_CUTOFF
    ) / total

    print(f"\nEvaluation: {label} | model {meta['model']} | {meta['chunks']} chunks | {total} questions\n")
    for r in rows:
        shown = r["rank"] if r["rank"] else "-"
        print(f"{r['id']}  rank {str(shown):>4}  {r['question'][:70]}")

    print("\nResults")
    for name, value in metrics.items():
        print(f"  {name:8s} {value:.0%}" if name != f"mrr@{MRR_CUTOFF}" else f"  {name:8s} {value:.3f}")

    misses = [r for r in rows if not r["rank"] or r["rank"] > 5]
    print(f"\nQuestions where the answer was not in the top 5: {len(misses)}")
    for r in misses:
        print(f"  {r['id']}  rank {r['rank'] if r['rank'] else '-'}  top result was {r['top_chunk']}")

    out_file = EVAL_DIR / f"results_{label}.json"
    out_file.write_text(json.dumps({
        "label": label,
        "model": meta["model"],
        "chunks": meta["chunks"],
        "metrics": metrics,
        "questions": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()