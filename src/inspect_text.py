import re
import sys
import json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EXTRACTED_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/extracted")
MIN_CHARS = 100


def load_pages(json_path):
    return json.loads(json_path.read_text(encoding="utf-8"))


def count_hyphen_breaks(pages):
    pattern = re.compile(r"[a-z]-[ \t]{0,5}\n[ \t]{0,5}[a-z]")
    return sum(len(pattern.findall(p["text"])) for p in pages)


def single_letter_ratio(pages):
    words = []
    for p in pages:
        words.extend(p["text"].split())
    if not words:
        return 0.0
    singles = [w for w in words if len(w) == 1 and w.lower() not in ("a", "i")]
    return len(singles) / len(words)


def find_repeated_lines(pages, min_share=0.3):
    counter = Counter()
    for p in pages:
        unique_lines = {line.strip() for line in p["text"].splitlines() if line.strip()}
        counter.update(unique_lines)
    threshold = max(2, int(len(pages) * min_share))
    return [(line, n) for line, n in counter.most_common(10) if n >= threshold]


def main():
    json_files = sorted(EXTRACTED_DIR.glob("*.json"))
    if not json_files:
        print(f"No extracted files found in {EXTRACTED_DIR}")
        return

    for json_path in json_files:
        pages = load_pages(json_path)
        print(f"\n===== {json_path.stem} =====")
        print(f"Words split by a hyphen at a line end: {count_hyphen_breaks(pages)}")
        print(f"Stray single-letter words: {single_letter_ratio(pages):.1%}")

        print("Lines repeated on many pages (likely headers/footers):")
        for line, count in find_repeated_lines(pages):
            print(f"  {count}x  {line[:80]}")

        for p in pages:
            if len(p["text"].strip()) < MIN_CHARS:
                print(f"--- low-text page {p['page']} ---")
                print(repr(p["text"][:300]))


if __name__ == "__main__":
    main()