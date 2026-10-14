import re
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
RRF_K = 60
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


def fuse_orders(embedding_order, bm25_order):
    fused = 1 / (RRF_K + rank_positions(embedding_order)) + 1 / (RRF_K + rank_positions(bm25_order))
    return np.argsort(fused)[::-1]


def load_retriever(strategy="sentences"):
    chunks = json.loads((CHUNKS_DIR / f"{strategy}.json").read_text(encoding="utf-8"))
    embeddings = np.load(INDEX_DIR / f"{strategy}_embeddings.npy")
    meta = json.loads((INDEX_DIR / f"{strategy}_meta.json").read_text(encoding="utf-8"))
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "meta": meta,
        "model": SentenceTransformer(meta["model"]),
        "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
    }


def retrieve(store, query, top_k=5):
    vector = store["model"].encode([query], normalize_embeddings=True)[0]
    cosine = store["embeddings"] @ vector
    bm25_scores = store["bm25"].get_scores(tokenize(query))
    order = fuse_orders(np.argsort(cosine)[::-1], np.argsort(bm25_scores)[::-1])

    hits = []
    for index in order[:top_k]:
        hits.append({
            "chunk": store["chunks"][int(index)],
            "cosine": float(cosine[index]),
            "bm25": float(bm25_scores[index]),
        })
    return {
        "hits": hits,
        "top_cosine": float(cosine.max()),
        "top_bm25": float(bm25_scores.max()),
    }