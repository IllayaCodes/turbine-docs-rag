from fastapi.testclient import TestClient

import api


def fake_answer(store, question):
    chunk = {"chunk_id": "doc_p4_c1", "source": "doc.pdf", "page": 4, "text": "Full passage text."}
    return {
        "query": question,
        "found": True,
        "top_cosine": 0.71234,
        "sentences": [("A quoted sentence from the report.", chunk, 0.9)],
        "hits": [{"chunk": chunk, "cosine": 0.71234, "bm25": 3.2}],
    }


def fake_not_found(store, question):
    chunk = {"chunk_id": "doc_p9_c2", "source": "doc.pdf", "page": 9, "text": "Unrelated passage."}
    return {
        "query": question,
        "found": False,
        "top_cosine": 0.16,
        "sentences": [],
        "hits": [{"chunk": chunk, "cosine": 0.16, "bm25": 0.0}],
    }


def test_health_reports_chunk_count(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": [1, 2, 3]})
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 3}


def test_ask_returns_answer_with_citation(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": []})
    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = TestClient(api.app).post("/ask", json={"question": "What is the pinch point?"})
    body = response.json()
    assert response.status_code == 200
    assert body["found"] is True
    assert body["best_score"] == 0.712
    assert body["answer_sentences"][0]["page"] == 4
    assert body["best_passage"] == "Full passage text."


def test_ask_says_not_found_without_an_answer(monkeypatch):
    monkeypatch.setitem(api.state, "store", {"chunks": []})
    monkeypatch.setattr(api, "answer_question", fake_not_found)
    response = TestClient(api.app).post("/ask", json={"question": "How do I bake bread?"})
    body = response.json()
    assert body["found"] is False
    assert body["answer_sentences"] == []
    assert body["best_passage"] is None


def test_ask_rejects_a_question_that_is_too_short():
    response = TestClient(api.app).post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_returns_503_when_the_index_is_not_loaded(monkeypatch):
    monkeypatch.delitem(api.state, "store", raising=False)
    response = TestClient(api.app).post("/ask", json={"question": "What is the pinch point?"})
    assert response.status_code == 503