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

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import UUID4, BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import db
from graph import MAX_QUERY_CHARS, agent_graph

# Our own loggers (finsight, finsight.resolver, finsight.collector) print INFO lines,
# e.g. "resolved 4 predictions"; uvicorn only configures its own loggers.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
for noisy in ("httpx", "apscheduler", "yfinance"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("finsight")


def start_scheduler() -> BackgroundScheduler | None:
    """Background jobs, only when predictions storage is on:
    - resolver every 15 min: grades predictions whose time is up
    - collector (opt-in, COLLECTOR_ENABLED=true): one engine run per asset each bar
    """
    if not db.enabled():
        return None
    import collector  # imported here: it pulls in the tools only when the jobs are wanted
    import resolver

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(resolver.resolve_due, "interval", minutes=15, id="resolver")
    if os.environ.get("COLLECTOR_ENABLED", "").lower() == "true":
        # a few minutes after each bar opens, so the new bar exists in the price data
        scheduler.add_job(collector.collect, "cron", minute=5, args=[["1H"]], id="collect_1h")
        scheduler.add_job(collector.collect, "cron", hour="0,4,8,12,16,20", minute=10, args=[["4H"]], id="collect_4h")
        scheduler.add_job(collector.collect, "cron", hour=0, minute=15, args=[["1D"]], id="collect_1d")
        scheduler.add_job(collector.collect, "cron", day_of_week="mon", hour=0, minute=20, args=[["1W"]], id="collect_1w")
    scheduler.start()
    log.info("scheduler started: %s", ", ".join(job.id for job in scheduler.get_jobs()))
    return scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    try:
        db.init_db()
        log.info("predictions storage: %s", "on" if db.enabled() else "off (DATABASE_URL not set)")
        scheduler = start_scheduler()
    except Exception:
        log.exception("could not initialise storage or the scheduler; continuing without them")
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(title="FinSight API", lifespan=lifespan)

# Per-IP rate limit on the endpoints that spend LLM tokens (denial-of-wallet protection).
# 429 + Retry-After when exceeded. In-memory: per server process, resets on restart.
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Only the frontend's own origin may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "http://localhost:8080").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"  # analyses are per-session; don't cache them in proxies
    return response

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
@limiter.limit(RATE_LIMIT)
def analyze(request: Request, response: Response, body: AnalyzeRequest):
    # `request`: the limiter reads the client IP from it; `response`: it adds the X-RateLimit headers to it
    session_id = str(body.session_id or uuid.uuid4())
    return _run({"messages": [HumanMessage(body.query)]}, session_id)


@app.post("/resume", response_model=AnalyzeResponse)
@limiter.limit(RATE_LIMIT)
def resume(request: Request, response: Response, body: ResumeRequest):
    session_id = str(body.session_id)
    state = agent_graph.get_state({"configurable": {"thread_id": session_id}})
    if not (state.tasks and state.tasks[0].interrupts):  # only a paused conversation can be resumed
        raise HTTPException(status_code=409, detail="This session is not waiting for an answer.")
    return _run(Command(resume=body.answer), session_id)


@app.get("/health")
def health():
    return {"status": "ok", "predictions_storage": db.enabled()}


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Log the details server-side; never send stack traces or internals to the client."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})
