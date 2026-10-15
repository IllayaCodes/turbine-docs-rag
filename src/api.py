from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ask import STRATEGY, answer_question
from retrieve import load_retriever

state = {}


@asynccontextmanager
async def lifespan(app):
    state["store"] = load_retriever(STRATEGY)
    yield
    state.clear()


app = FastAPI(title="turbine-docs-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def format_response(response):
    hits = response["hits"]
    shown = hits if response["found"] else hits[:1]
    return {
        "question": response["query"],
        "found": response["found"],
        "best_score": round(response["top_cosine"], 3),
        "answer_sentences": [
            {"text": sentence, "source": chunk["source"], "page": chunk["page"]}
            for sentence, chunk, _ in response["sentences"]
        ],
        "best_passage": hits[0]["chunk"]["text"] if response["found"] and hits else None,
        "sources": [
            {
                "chunk_id": hit["chunk"]["chunk_id"],
                "source": hit["chunk"]["source"],
                "page": hit["chunk"]["page"],
                "score": round(hit["cosine"], 3),
            }
            for hit in shown
        ],
    }


@app.get("/health")
def health():
    store = state.get("store")
    if store is None:
        return {"status": "starting", "chunks": 0}
    return {"status": "ok", "chunks": len(store["chunks"])}


@app.post("/ask")
def ask(request: AskRequest):
    store = state.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="The index is not loaded yet")
    return format_response(answer_question(store, request.question))