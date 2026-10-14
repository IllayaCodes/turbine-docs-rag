import re
import sys

import numpy as np

from retrieve import load_retriever, retrieve, tokenize

sys.stdout.reconfigure(encoding="utf-8")

STRATEGY = "sentences"
NOT_FOUND_THRESHOLD = 0.40
CHUNKS_TO_READ = 3
SENTENCES_IN_ANSWER = 2
MIN_SENTENCE_WORDS = 6
OVERLAP_WEIGHT = 0.5
PASSAGE_CHARS = 700

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text):
    sentences = []
    for line in text.splitlines():
        for part in SENTENCE_END.split(line.strip()):
            part = part.strip()
            if len(part.split()) >= MIN_SENTENCE_WORDS:
                sentences.append(part)
    return sentences


def candidate_sentences(hits):
    candidates = []
    seen = set()
    for hit in hits[:CHUNKS_TO_READ]:
        for sentence in split_sentences(hit["chunk"]["text"]):
            if sentence not in seen:
                seen.add(sentence)
                candidates.append((sentence, hit["chunk"]))
    return candidates


def overlap_score(store, query_tokens, sentence):
    idf = store["bm25"].idf
    wanted = set(query_tokens)
    total = sum(idf.get(t, 0.0) for t in wanted)
    if total <= 0:
        return 0.0
    present = set(tokenize(sentence))
    return sum(idf.get(t, 0.0) for t in wanted if t in present) / total


def score_candidates(store, query, candidates):
    model = store["model"]
    vectors = model.encode([s for s, _ in candidates], normalize_embeddings=True)
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    cosine = vectors @ query_vector
    query_tokens = tokenize(query)
    overlap = np.array([overlap_score(store, query_tokens, s) for s, _ in candidates])
    return cosine, overlap


def choose_sentences(candidates, cosine, overlap, weight):
    scores = cosine + np.multiply(weight, overlap)
    best = np.argsort(-scores, kind="stable")[:SENTENCES_IN_ANSWER]
    return [(candidates[i][0], candidates[i][1], float(scores[i])) for i in best]


def pick_sentences(store, query, hits, weight=OVERLAP_WEIGHT):
    candidates = candidate_sentences(hits)
    if not candidates:
        return []
    cosine, overlap = score_candidates(store, query, candidates)
    return choose_sentences(candidates, cosine, overlap, weight)


def answer_question(store, query):
    result = retrieve(store, query)
    found = result["top_cosine"] >= NOT_FOUND_THRESHOLD
    sentences = pick_sentences(store, query, result["hits"]) if found else []
    return {
        "query": query,
        "found": found,
        "top_cosine": result["top_cosine"],
        "sentences": sentences,
        "hits": result["hits"],
    }


def show(response):
    print(f"\nQuestion: {response['query']}")
    if not response["found"]:
        print(
            f"Not found in the documents (best match score {response['top_cosine']:.2f}, "
            f"threshold {NOT_FOUND_THRESHOLD:.2f})."
        )
        closest = response["hits"][0]["chunk"]
        print(f"Closest passage, not an answer: [{closest['source']}, page {closest['page']}]")
        return

    print(f"\nAnswer (sentences quoted from the documents; best match score {response['top_cosine']:.2f}):")
    for sentence, chunk, score in response["sentences"]:
        print(f'  - "{sentence}"  [{chunk["source"]}, page {chunk["page"]}]')

    best = response["hits"][0]["chunk"]
    print(f"\nBest passage [{best['source']}, page {best['page']}]:")
    print(f"  {best['text'][:PASSAGE_CHARS]}")

    print("\nTop sources:")
    for hit in response["hits"]:
        chunk = hit["chunk"]
        print(f"  {chunk['chunk_id']}  [{chunk['source']}, page {chunk['page']}]")


def main():
    store = load_retriever(STRATEGY)
    if len(sys.argv) > 1:
        show(answer_question(store, " ".join(sys.argv[1:])))
        return

    print("Ready. Type a question, or press Enter on an empty line to quit.")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            break
        show(answer_question(store, query))


if __name__ == "__main__":
    main()