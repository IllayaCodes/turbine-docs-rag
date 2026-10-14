import sys
import json
from pathlib import Path

import numpy as np

from retrieve import load_retriever, retrieve

sys.stdout.reconfigure(encoding="utf-8")

STRATEGY = "sentences"
EVAL_DIR = Path("data/eval")
ANSWERABLE_FILES = ("questions.json", "questions_paraphrased.json")
UNANSWERABLE_FILE = "questions_unanswerable.json"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def load_questions(name):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def describe(label, scores):
    low, q1, mid, q3, high = np.percentile(scores, [0, 25, 50, 75, 100])
    print(f"{label:32s} n={len(scores):3d}  min {low:.2f}  25% {q1:.2f}  median {mid:.2f}  75% {q3:.2f}  max {high:.2f}")


def main():
    store = load_retriever(STRATEGY)

    answerable = []
    for name in ANSWERABLE_FILES:
        for question in load_questions(name):
            answerable.append(retrieve(store, question["question"], top_k=1)["top_cosine"])

    groups = {}
    for question in load_questions(UNANSWERABLE_FILE):
        score = retrieve(store, question["question"], top_k=1)["top_cosine"]
        groups.setdefault(question["category"], []).append(score)
    unanswerable = [s for scores in groups.values() for s in scores]

    print("\nBest embedding score per question (higher means a closer match)\n")
    describe("answerable questions", answerable)
    for category, scores in groups.items():
        describe(f"unanswerable: {category}", scores)

    print("\nIf the threshold is T, a question is answered when its best score is at least T.\n")
    header = "T      answerable kept   unanswerable rejected"
    for category in groups:
        header += f"   {category} rejected"
    print(header)

    rows = []
    best = None
    for t in THRESHOLDS:
        kept = float(np.mean([s >= t for s in answerable]))
        rejected = float(np.mean([s < t for s in unanswerable]))
        per_group = {c: float(np.mean([s < t for s in scores])) for c, scores in groups.items()}
        balance = (kept + rejected) / 2
        rows.append({"threshold": t, "answerable_kept": kept, "unanswerable_rejected": rejected, "per_category": per_group})
        line = f"{t:.2f}   {kept:15.0%}   {rejected:21.0%}"
        for c in groups:
            line += f"   {per_group[c]:>{len(c) + 9}.0%}"
        print(line)
        if best is None or balance > best[1]:
            best = (t, balance)

    print(f"\nHighest average of the two columns: T = {best[0]:.2f}")
    print("Pick T from the table yourself. Raising T rejects more bad questions but also loses real ones.")

    out_file = EVAL_DIR / "threshold_report.json"
    out_file.write_text(json.dumps({"rows": rows, "suggested": best[0]}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()