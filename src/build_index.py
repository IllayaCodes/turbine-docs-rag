import sys
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sentences"
    chunks_file = CHUNKS_DIR / f"{strategy}.json"
    if not chunks_file.exists():
        print(f"Missing {chunks_file}. Run: python src\\chunk_text.py {strategy}")
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    print(f"Loading model {MODEL_NAME} (the first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    limit = model.max_seq_length
    encoded = model.tokenizer(texts, truncation=False)["input_ids"]
    token_counts = [len(ids) for ids in encoded]
    too_long = sum(1 for n in token_counts if n > limit)
    print(f"The model reads at most {limit} tokens per chunk")
    print(f"Chunks longer than that: {too_long} of {len(texts)} (longest: {max(token_counts)} tokens)")

    long_ids = [chunks[i]["chunk_id"] for i, n in enumerate(token_counts) if n > limit]
    print("Too-long chunks:", ", ".join(long_ids))

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / f"{strategy}_embeddings.npy", embeddings)
    meta = {
        "model": MODEL_NAME,
        "strategy": strategy,
        "chunks": len(chunks),
        "dimensions": int(embeddings.shape[1]),
    }
    (INDEX_DIR / f"{strategy}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nEmbedded {len(chunks)} chunks into {embeddings.shape[1]} numbers each")
    print(f"Saved -> {INDEX_DIR / (strategy + '_embeddings.npy')}")


if __name__ == "__main__":
    main()