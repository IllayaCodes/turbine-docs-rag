import re
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

IN_DIR = Path("data/cleaned")
OUT_DIR = Path("data/chunks")
CHUNK_WORDS = 150
OVERLAP_WORDS = 30
MIN_CHUNK_WORDS = 15
STRATEGIES = ("sentences", "words")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def chunk_words(text, size, overlap):
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def split_units(text, max_words):
    units = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in SENTENCE_END.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence.split()) > max_words:
                units.extend(chunk_words(sentence, max_words, 0))
            else:
                units.append(sentence)
    return units


def chunk_sentences(text, size, overlap):
    units = split_units(text, size)
    chunks = []
    current = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > size:
            chunks.append(" ".join(current))
            carry = []
            carry_count = 0
            for earlier in reversed(current):
                w = len(earlier.split())
                if carry_count + w > overlap:
                    break
                carry.insert(0, earlier)
                carry_count += w
            current = carry
            count = carry_count
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(pages, strategy):
    splitter = chunk_sentences if strategy == "sentences" else chunk_words
    results = []
    for page in pages:
        pieces = splitter(page["text"], CHUNK_WORDS, OVERLAP_WORDS)
        for index, piece in enumerate(pieces, start=1):
            if len(piece.split()) < MIN_CHUNK_WORDS:
                continue
            stem = Path(page["source"]).stem
            results.append({
                "chunk_id": f"{stem}_p{page['page']}_c{index}",
                "source": page["source"],
                "page": page["page"],
                "text": piece,
                "word_count": len(piece.split()),
            })
    return results


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sentences"
    if strategy not in STRATEGIES:
        print(f"Unknown strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")
        return

    json_files = sorted(IN_DIR.glob("*.json"))
    if not json_files:
        print(f"No cleaned files found in {IN_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for json_path in json_files:
        pages = json.loads(json_path.read_text(encoding="utf-8"))
        chunks = build_chunks(pages, strategy)
        counts = [c["word_count"] for c in chunks]
        print(f"{json_path.stem}: {len(chunks)} chunks | words per chunk: "
              f"min {min(counts)}, avg {sum(counts) // len(counts)}, max {max(counts)}")
        all_chunks.extend(chunks)

    out_file = OUT_DIR / f"{strategy}.json"
    out_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    example = all_chunks[len(all_chunks) // 2]
    print(f"\nStrategy: {strategy} | size {CHUNK_WORDS} | overlap {OVERLAP_WORDS}")
    print(f"Total chunks: {len(all_chunks)} -> {out_file}")
    print(f"\n--- example chunk: {example['chunk_id']} ({example['source']}, page {example['page']}) ---")
    print(example["text"])


if __name__ == "__main__":
    main()