import re
import sys
import json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

IN_DIR = Path("data/extracted")
OUT_DIR = Path("data/cleaned")
MIN_CHARS = 100
EDGE_LINES = 3
MIN_PAGE_SHARE = 0.4
MIN_PAGES_FOR_PATTERN = 3

HYPHEN_BREAK = re.compile(r"([a-z])-[ \t]{0,5}\n[ \t]{0,5}([a-z])")
SOFT_BREAK = re.compile(r"\n(?=[a-z])")
EXTRA_SPACES = re.compile(r"[ \t]{2,}")
EXTRA_NEWLINES = re.compile(r"\n{3,}")


def normalize(line):
    return re.sub(r"\d+", "0", line.strip())


def edge_indices(lines):
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:EDGE_LINES] + filled[-EDGE_LINES:])


def find_header_footer_patterns(pages):
    counter = Counter()
    for p in pages:
        lines = p["text"].splitlines()
        shapes = {normalize(lines[i]) for i in edge_indices(lines)}
        counter.update(shapes)
    needed = max(MIN_PAGES_FOR_PATTERN, int(len(pages) * MIN_PAGE_SHARE))
    return {shape for shape, count in counter.items() if count >= needed}


def remove_headers_footers(text, patterns):
    lines = text.splitlines()
    edges = edge_indices(lines)
    kept = []
    removed = 0
    for i, line in enumerate(lines):
        if i in edges and normalize(line) in patterns:
            removed += 1
        else:
            kept.append(line.rstrip())
    return "\n".join(kept), removed


def clean_page_text(text, patterns):
    text, removed = remove_headers_footers(text, patterns)
    joined = len(HYPHEN_BREAK.findall(text))
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = SOFT_BREAK.sub(" ", text)
    text = EXTRA_SPACES.sub(" ", text)
    text = EXTRA_NEWLINES.sub("\n\n", text)
    return text.strip(), removed, joined


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(IN_DIR.glob("*.json"))
    if not json_files:
        print(f"No extracted files found in {IN_DIR}")
        return

    for json_path in json_files:
        pages = json.loads(json_path.read_text(encoding="utf-8"))
        patterns = find_header_footer_patterns(pages)

        cleaned = []
        dropped = []
        total_removed = 0
        total_joined = 0
        for p in pages:
            text, removed, joined = clean_page_text(p["text"], patterns)
            total_removed += removed
            total_joined += joined
            if len(text) < MIN_CHARS:
                dropped.append(p["page"])
                continue
            cleaned.append({"source": p["source"], "page": p["page"], "text": text})

        chars_before = sum(len(p["text"]) for p in pages)
        chars_after = sum(len(p["text"]) for p in cleaned)

        print(f"\n===== {json_path.stem} =====")
        print(f"Header/footer shapes found: {sorted(patterns)}")
        print(f"Header/footer lines removed: {total_removed}")
        print(f"Split words joined: {total_joined}")
        print(f"Pages dropped (too little text): {dropped if dropped else 'none'}")
        print(f"Pages kept: {len(cleaned)} | Characters: {chars_before} -> {chars_after}")

        out_file = OUT_DIR / f"{json_path.stem}.json"
        out_file.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()