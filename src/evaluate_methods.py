import re
import sys
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from search import load_index
from evaluate import is_relevant, rank_of_first_relevant

sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"
CUTOFFS = (1, 3, 5, 10)
MRR_CUTOFF = 10
RRF_K = 60
METHODS = ("embeddings", "bm25", "hybrid")
STOPWORDS = set(
    "a an the of to in on for and or is are was were be by with what how which "
    "does did do at as it its that this from".split()
)


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def rank_positions(order):
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(1, len(order) + 1)
    return positions


def hybrid_order(embedding_order, bm25_order):
    fused = 1 / (RRF_K + rank_positions(embedding_order)) + 1 / (RRF_K + rank_positions(bm25_order))
    return np.argsort(fused)[::-1]


def summarize(ranks):
    total = len(ranks)
    metrics = {}
    for k in CUTOFFS:
        metrics[f"hit@{k}"] = sum(1 for r in ranks if r and r <= k) / total
    metrics[f"mrr@{MRR_CUTOFF}"] = sum(1 / r for r in ranks if r and r <= MRR_CUTOFF) / total
    return metrics


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sentences"
    label = sys.argv[2] if len(sys.argv) > 2 else strategy + "_methods"

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    chunks, embeddings, meta = load_index(strategy)
    model = SentenceTransformer(meta["model"])
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    vectors = model.encode([q["question"] for q in questions], normalize_embeddings=True)

    ranks = {name: [] for name in METHODS}
    for question, vector in zip(questions, vectors):
        embedding_order = np.argsort(embeddings @ vector)[::-1]
        bm25_order = np.argsort(bm25.get_scores(tokenize(question["question"])))[::-1]
        orders = {
            "embeddings": embedding_order,
            "bm25": bm25_order,
            "hybrid": hybrid_order(embedding_order, bm25_order),
        }
        for name in METHODS:
            ranks[name].append(rank_of_first_relevant(orders[name], chunks, question))

    def show(value):
        return str(value) if value else "-"

    print(f"\nComparison: {strategy} chunks ({meta['chunks']}) | model {meta['model']} | {len(questions)} questions\n")
    print("id    embeddings  bm25  hybrid  question")
    for i, question in enumerate(questions):
        print(
            f"{question['id']}  {show(ranks['embeddings'][i]):>10}  {show(ranks['bm25'][i]):>4}  "
            f"{show(ranks['hybrid'][i]):>6}  {question['question'][:55]}"
        )

    summaries = {name: summarize(ranks[name]) for name in METHODS}
    print("\nResults")
    print(f"{'method':12s} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7} {'mrr@10':>7}")
    for name in METHODS:
        m = summaries[name]
        print(
            f"{name:12s} {m['hit@1']:>6.0%} {m['hit@3']:>6.0%} {m['hit@5']:>6.0%} "
            f"{m['hit@10']:>7.0%} {m['mrr@10']:>7.3f}"
        )

    for name in METHODS:
        missed = [questions[i]["id"] + f" (rank {show(r)})" for i, r in enumerate(ranks[name]) if not r or r > 5]
        print(f"\n{name}: not in top 5 -> {', '.join(missed) if missed else 'none'}")

    out_file = EVAL_DIR / f"results_{label}.json"
    out_file.write_text(json.dumps({
        "label": label,
        "strategy": strategy,
        "model": meta["model"],
        "chunks": meta["chunks"],
        "metrics": summaries,
        "ranks": {name: {questions[i]["id"]: r for i, r in enumerate(ranks[name])} for name in METHODS},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()