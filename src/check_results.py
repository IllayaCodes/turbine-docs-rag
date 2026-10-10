import json
from pathlib import Path

EVAL_DIR = Path("data/eval")
TOTAL = 40

EXPECTED = {
    "sentences_methods": {
        "embeddings": {"hit@1": 19, "hit@3": 30, "hit@5": 36, "hit@10": 38, "mrr@10": 0.648},
        "bm25": {"hit@1": 31, "hit@3": 36, "hit@5": 39, "hit@10": 39, "mrr@10": 0.843},
    },
    "words_methods": {
        "embeddings": {"hit@1": 19, "hit@3": 29, "hit@5": 33, "hit@10": 38, "mrr@10": 0.624},
        "bm25": {"hit@1": 29, "hit@3": 35, "hit@5": 39, "hit@10": 39, "mrr@10": 0.804},
    },
}
ALLOWED_QUESTION_DIFF = {"embeddings": 0, "bm25": 1}
ALLOWED_MRR_DIFF = {"embeddings": 0.002, "bm25": 0.01}


def main():
    problems = 0
    for label, methods in EXPECTED.items():
        path = EVAL_DIR / f"results_{label}.json"
        if not path.exists():
            print(f"\n{label}: file not found ({path}). Run evaluate_methods.py first.")
            problems += 1
            continue
        results = json.loads(path.read_text(encoding="utf-8"))["metrics"]

        print(f"\n===== {label} =====")
        for method, expected in methods.items():
            print(f"{method}")
            for name, want in expected.items():
                got = results[method][name]
                if name == "mrr@10":
                    ok = abs(got - want) <= ALLOWED_MRR_DIFF[method]
                    print(f"  {name:7s} expected {want:.3f} | yours {got:.3f} | {'OK' if ok else 'DIFFERENT'}")
                else:
                    got_count = round(got * TOTAL)
                    ok = abs(got_count - want) <= ALLOWED_QUESTION_DIFF[method]
                    print(f"  {name:7s} expected {want:2d} of {TOTAL} | yours {got_count:2d} of {TOTAL} | {'OK' if ok else 'DIFFERENT'}")
                if not ok:
                    problems += 1

    print()
    if problems == 0:
        print("All checked numbers match. Hybrid has no expected value, so read that row yourself.")
    else:
        print(f"{problems} number(s) differ. Stop here and send me the full output of evaluate_methods.py.")


if __name__ == "__main__":
    main()