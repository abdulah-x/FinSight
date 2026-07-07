"""FastAPI app exposing the FinSight research graph."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import research_graph

app = FastAPI(title="FinSight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


@app.post("/research")
def research(request: ResearchRequest):
    result = research_graph.invoke({"query": request.query})
    if result.get("error"):
        return {"error": result["error"]}
    return {
        "ticker": result["ticker"],
        "market_data": result["market_data"],
        "report": result["report"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
