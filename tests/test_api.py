from fastapi.testclient import TestClient

import api


def make_chunk(number, page):
    return {
        "chunk_id": f"doc_p{page}_c{number}",
        "source": "doc.pdf",
        "page": page,
        "text": f"Passage number {number}.",
    }


def fake_answer(store, question):
    first = make_chunk(1, 4)
    second = make_chunk(2, 5)
    return {
        "query": question,
        "found": True,
        "top_cosine": 0.71234,
        "sentences": [("A quoted sentence from the report.", first, 0.9)],
        "hits": [
            {"chunk": first, "cosine": 0.71234, "bm25": 3.2},
            {"chunk": second, "cosine": 0.55, "bm25": 1.1},
        ],
    }


def fake_not_found(store, question):
    first = make_chunk(1, 9)
    second = make_chunk(2, 10)
    return {
        "query": question,
        "found": False,
        "top_cosine": 0.16,
        "sentences": [],
        "hits": [
            {"chunk": first, "cosine": 0.16, "bm25": 0.0},
            {"chunk": second, "cosine": 0.10, "bm25": 0.0},
        ],
    }


def test_health_reports_chunk_count(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": [1, 2, 3]})
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 3}


def test_ask_returns_answer_with_citation_and_all_sources(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": []})
    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = TestClient(api.app).post("/ask", json={"question": "What is the pinch point?"})
    body = response.json()
    assert response.status_code == 200
    assert body["found"] is True
    assert body["best_score"] == 0.712
    assert body["answer_sentences"][0]["page"] == 4
    assert body["best_passage"] == "Passage number 1."
    assert len(body["sources"]) == 2


def test_ask_says_not_found_and_lists_only_the_closest_source(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": []})
    monkeypatch.setattr(api, "answer_question", fake_not_found)
    response = TestClient(api.app).post("/ask", json={"question": "How do I bake bread?"})
    body = response.json()
    assert body["found"] is False
    assert body["answer_sentences"] == []
    assert body["best_passage"] is None
    assert len(body["sources"]) == 1
    assert body["sources"][0]["chunk_id"] == "doc_p9_c1"


def test_ask_rejects_a_question_that_is_too_short():
    response = TestClient(api.app).post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_returns_503_when_the_index_is_not_loaded(monkeypatch):
    monkeypatch.delitem(api.state, "store", raising=False)
    response = TestClient(api.app).post("/ask", json={"question": "What is the pinch point?"})
    assert response.status_code == 503