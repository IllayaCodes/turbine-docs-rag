import numpy as np

from retrieve import fuse_orders, rank_positions, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Pinch Point?") == ["pinch", "point"]


def test_rank_positions_gives_rank_one_to_first_item_of_order():
    positions = rank_positions(np.array([2, 0, 1]))
    assert positions.tolist() == [2, 3, 1]


def test_fuse_orders_puts_chunk_that_wins_both_rankings_first():
    embedding_order = np.array([3, 1, 0, 2])
    bm25_order = np.array([3, 0, 2, 1])
    fused = fuse_orders(embedding_order, bm25_order)
    assert fused[0] == 3


def test_fuse_orders_returns_every_chunk_once():
    fused = fuse_orders(np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0]))
    assert sorted(fused.tolist()) == [0, 1, 2, 3]