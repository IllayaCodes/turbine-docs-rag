from types import SimpleNamespace

import numpy as np

from ask import (
    candidate_sentences,
    choose_sentences,
    overlap_score,
    split_sentences,
)


def test_split_sentences_drops_short_fragments():
    text = "This sentence has more than six words in it. Too short."
    assert split_sentences(text) == ["This sentence has more than six words in it."]


def test_overlap_score_counts_rare_words_more():
    store = {"bm25": SimpleNamespace(idf={"pinch": 5.0, "tip": 1.0})}
    with_rare = overlap_score(store, ["pinch", "tip"], "The pinch occurs during acceleration.")
    with_common = overlap_score(store, ["pinch", "tip"], "The tip moves during acceleration.")
    assert with_rare > with_common
    assert round(with_rare, 3) == round(5.0 / 6.0, 3)


def test_overlap_score_is_zero_when_query_words_are_unknown():
    store = {"bm25": SimpleNamespace(idf={})}
    assert overlap_score(store, ["anything"], "Some sentence about turbines here.") == 0.0


def test_choose_sentences_prefers_higher_combined_score():
    candidates = [("first sentence here", {}), ("second sentence here", {}), ("third sentence here", {})]
    cosine = np.array([0.50, 0.40, 0.30])
    overlap = np.array([0.0, 0.9, 0.0])
    picked = choose_sentences(candidates, cosine, overlap, weight=0.5)
    assert picked[0][0] == "second sentence here"


def test_candidate_sentences_removes_duplicates_across_chunks():
    sentence = "This exact sentence appears in two overlapping chunks."
    hits = [
        {"chunk": {"text": sentence, "chunk_id": "a"}},
        {"chunk": {"text": sentence, "chunk_id": "b"}},
    ]
    found = candidate_sentences(hits)
    assert len(found) == 1