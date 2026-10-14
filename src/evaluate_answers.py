import sys
import json
from pathlib import Path

from retrieve import load_retriever, retrieve
from ask import (
    OVERLAP_WEIGHT,
    candidate_sentences,
    score_candidates,
    choose_sentences,
)

sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path("data/eval")
QUESTION_FILES = ("questions.json", "questions_paraphrased.json")
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]


def normalize(text):
    return " ".join(text.lower().split())


def contains_answer(sentence, chunk, question):
    if not chunk["source"].startswith(question["document"]):
        return False
    text = normalize(sentence)
    return any(normalize(phrase) in text for phrase in question["answer_phrases"])


def main():
    store = load_retriever("sentences")
    questions = []
    for name in QUESTION_FILES:
        questions.extend(json.loads((EVAL_DIR / name).read_text(encoding="utf-8")))
    total = len(questions)

    available = 0
    hits = {w: 0 for w in WEIGHTS}
    missed_at_default = []

    for question in questions:
        result = retrieve(store, question["question"])
        candidates = candidate_sentences(result["hits"])
        if not candidates:
            missed_at_default.append(question["id"])
            continue

        if any(contains_answer(s, c, question) for s, c in candidates):
            available += 1

        cosine, overlap = score_candidates(store, question["question"], candidates)
        for weight in WEIGHTS:
            picked = choose_sentences(candidates, cosine, overlap, weight)
            if any(contains_answer(s, c, question) for s, c, _ in picked):
                hits[weight] += 1
            elif weight == OVERLAP_WEIGHT:
                missed_at_default.append(question["id"])

    print(f"\nQuestions: {total}")
    print(f"Answer sentence exists among the sentences of the top 3 chunks (upper limit): {available} of {total}\n")
    print("weight   answer in the 2 picked sentences")
    for weight in WEIGHTS:
        label = "  (embedding only)" if weight == 0.0 else ("  (current setting)" if weight == OVERLAP_WEIGHT else "")
        print(f"{weight:5.2f}    {hits[weight]:2d} of {total}  ({hits[weight] / total:.0%}){label}")

    print(f"\nMissed at the current setting ({OVERLAP_WEIGHT}): {', '.join(missed_at_default) if missed_at_default else 'none'}")


if __name__ == "__main__":
    main()