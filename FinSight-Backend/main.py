"""FastAPI app exposing the FinSight bias agent.

    POST /analyze  {query, session_id?}  -> complete | needs_input | refused
    POST /resume   {session_id, answer}  -> continues a conversation paused by ask_user
    GET  /health
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import UUID4, BaseModel, Field

import db
from graph import MAX_QUERY_CHARS, agent_graph

log = logging.getLogger("finsight")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db.init_db()
        log.info("predictions storage: %s", "on" if db.enabled() else "off (DATABASE_URL not set)")
    except Exception:
        log.exception("could not initialise the predictions table; continuing without storage")
    yield


app = FastAPI(title="FinSight API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened to the frontend origin in Stage 6.5
    allow_methods=["*"],
    allow_headers=["*"],
)

# Langfuse tracing is optional: only attached when its keys are configured.
callbacks = []
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    from langfuse.langchain import CallbackHandler

    callbacks.append(CallbackHandler())


# ---------- request / response models ----------

class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    session_id: UUID4 | None = None  # one conversation; omit to start a new one


class ResumeRequest(BaseModel):
    session_id: UUID4
    answer: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class Analysis(BaseModel):
    asset: str
    timeframe: str
    bias: Literal["Bullish", "Bearish", "Neutral"]
    score: int
    bucket: Literal["Low", "Medium", "High"]
    coverage: float
    event_penalty: float
    breakdown: list[str]


class AnalyzeResponse(BaseModel):
    status: Literal["complete", "needs_input", "refused"]
    session_id: str
    answer: str | None = None      # the full text answer (complete / refused)
    question: str | None = None    # what the agent asks (needs_input)
    analysis: Analysis | None = None  # structured bias + score, when a new analysis was made
    cached: bool = False
    stop_reason: str | None = None   # which cap stopped the agent, if any


# ---------- endpoints ----------

def _run(graph_input, session_id: str) -> AnalyzeResponse:
    config = {"configurable": {"thread_id": session_id}, "callbacks": callbacks}
    result = agent_graph.invoke(graph_input, config=config)

    paused = agent_graph.get_state(config)
    if paused.tasks and paused.tasks[0].interrupts:
        question = paused.tasks[0].interrupts[0].value["question"]
        return AnalyzeResponse(status="needs_input", session_id=session_id, question=question)

    stop_reason = result.get("stop_reason")
    if stop_reason and stop_reason.startswith("LLM unavailable"):
        raise HTTPException(status_code=503, detail="The analysis service is busy. Try again in a minute.",
                            headers={"Retry-After": "60"})

    analysis, cached = result.get("analysis"), bool(result.get("cache_hit"))
    if analysis and not cached:  # a cached answer was already stored when it was first made
        try:
            db.save_prediction(analysis)
        except Exception:
            log.exception("could not store prediction")  # never fail the trader's request over it

    return AnalyzeResponse(
        status="refused" if result.get("guardrail_result") != "ok" else "complete",
        session_id=session_id,
        answer=result["messages"][-1].content,
        analysis=Analysis(**analysis) if analysis else None,
        cached=cached,
        stop_reason=stop_reason,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    session_id = str(request.session_id or uuid.uuid4())
    return _run({"messages": [HumanMessage(request.query)]}, session_id)


@app.post("/resume", response_model=AnalyzeResponse)
def resume(request: ResumeRequest):
    session_id = str(request.session_id)
    state = agent_graph.get_state({"configurable": {"thread_id": session_id}})
    if not (state.tasks and state.tasks[0].interrupts):  # only a paused conversation can be resumed
        raise HTTPException(status_code=409, detail="This session is not waiting for an answer.")
    return _run(Command(resume=request.answer), session_id)


@app.get("/health")
def health():
    return {"status": "ok", "predictions_storage": db.enabled()}


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Log the details server-side; never send stack traces or internals to the client."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})
