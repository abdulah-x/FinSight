"""FastAPI app exposing the FinSight research graph."""

import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from graph import research_graph

app = FastAPI(title="FinSight API")

langfuse_handler = CallbackHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/research")
def research(request: ResearchRequest):
    session_id = request.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [langfuse_handler],
    }
    result = research_graph.invoke({"query": request.query}, config=config)

    if result.get("error"):
        return {"error": result["error"], "session_id": session_id}

    return {
        "session_id": session_id,
        "intent": result.get("intent"),
        "ticker": result.get("ticker"),
        "market_data": result.get("market_data"),
        "social_posts": result.get("social_posts", []),
        "report": result["report"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
