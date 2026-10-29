"""FastAPI app exposing the FinSight bias agent.

    POST /analyze  {query, session_id?}  -> complete | needs_input | refused
    POST /resume   {session_id, answer}  -> continues a conversation paused by ask_user
    POST /analyze/stream, /resume/stream  -> the same, streamed as server-sent events:
         one event per agent step (for the page's live run log), then the result
    GET  /markets  -> latest price, 24h change and sparkline per asset, plus the next
                      high-impact event (no LLM; cached 60s)
    GET  /health
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import UUID4, BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import yfinance as yf

import db
from graph import MAX_QUERY_CHARS, agent_graph
from tools import get_economic_calendar

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

    # APScheduler skips a run that starts >1s late by default; Docker on a laptop wakes jobs
    # a few seconds late (a resolver run was lost this way), so allow 5 min and merge backlogs
    scheduler = BackgroundScheduler(timezone="UTC", job_defaults={"misfire_grace_time": 300, "coalesce": True})
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


class Row(BaseModel):
    group: str                 # Price, Macro, News, Positioning, Sentiment
    weight: float              # share of the score, 0..1
    value: float | None        # -1 bearish .. +1 bullish; None = no evidence this time
    label: str                 # "weak bullish", "missing", ...
    detail: str | None


class Analysis(BaseModel):
    asset: str
    timeframe: str
    bias: Literal["Bullish", "Bearish", "Neutral"]
    score: int
    bucket: Literal["Low", "Medium", "High"]
    net: float                 # where the evidence lands, -1..+1 (Neutral inside +-0.15)
    coverage: float
    event_penalty: float
    breakdown: list[str]
    rows: list[Row] = []


class Source(BaseModel):
    name: str                  # price_action, macro, calendar, ...
    ok: bool                   # False = the source failed or timed out


class Trace(BaseModel):
    """What the agent did this turn, shown in the page's run log."""
    sources: list[Source] = []
    tool_rounds: int = 0
    follow_up: bool = False    # answered from the previous result, no new evidence


class AnalyzeResponse(BaseModel):
    status: Literal["complete", "needs_input", "refused"]
    session_id: str
    answer: str | None = None      # the full text answer (complete / refused)
    question: str | None = None    # what the agent asks (needs_input)
    analysis: Analysis | None = None  # structured bias + score, when a new analysis was made
    cached: bool = False
    stop_reason: str | None = None   # which cap stopped the agent, if any
    trace: Trace = Trace()


# ---------- endpoints ----------

def _trace(result: dict) -> Trace:
    """Sources the agent fetched this turn (from the tool results' raw copies)."""
    messages = result.get("messages", [])
    turn_start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    sources = {}
    for m in messages[turn_start:]:
        if isinstance(m, ToolMessage) and m.name == "gather_evidence" and m.artifact:
            output = m.artifact["output"]
            if "error" not in output:  # the whole call failed (e.g. invalid arguments)
                sources.update({name: "error" not in out for name, out in output.items()})
    return Trace(sources=[Source(name=n, ok=ok) for n, ok in sources.items()],
                 tool_rounds=result.get("tool_rounds") or 0, follow_up=bool(result.get("follow_up")))


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}, "callbacks": callbacks}


def _run(graph_input, session_id: str) -> AnalyzeResponse:
    config = _config(session_id)
    return _finish(agent_graph.invoke(graph_input, config=config), session_id, config)


def _finish(result: dict, session_id: str, config: dict) -> AnalyzeResponse:
    """Final state -> response (shared by the plain and the streamed endpoints)."""
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
        trace=_trace(result),
    )


def _events(update: dict):
    """One finished graph step -> run-log events for the page. Facts only; the page words them."""
    for node, out in update.items():
        if not isinstance(out, dict):  # e.g. LangGraph's own "__interrupt__" marker
            continue
        if node == "guardrail":
            yield {"event": "checked", "category": out.get("guardrail_result"), "asset": out.get("asset"),
                   "timeframe": out.get("timeframe"), "cached": bool(out.get("cache_hit")),
                   "follow_up": bool(out.get("follow_up"))}
        elif node == "agent":
            message = out["messages"][-1]
            for call in message.tool_calls:
                if call["name"] == "gather_evidence":
                    yield {"event": "planned", "sources": call["args"].get("sources", [])}
                elif call["name"] == "ask_user":
                    yield {"event": "asking"}
            if not message.tool_calls:
                yield {"event": "agent_done"}  # next: finalize scores and writes, or the turn ends
        elif node == "tool_executor":
            sources = {}
            for m in out["messages"]:
                if m.name == "gather_evidence" and m.artifact and "error" not in m.artifact["output"]:
                    sources.update({name: "error" not in res for name, res in m.artifact["output"].items()})
            if sources:
                yield {"event": "gathered", "sources": [{"name": n, "ok": ok} for n, ok in sources.items()]}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _stream(graph_input, session_id: str) -> StreamingResponse:
    config = _config(session_id)

    def events():
        try:
            for update in agent_graph.stream(graph_input, config=config, stream_mode="updates"):
                for event in _events(update):
                    yield _sse(event)
            response = _finish(agent_graph.get_state(config).values, session_id, config)
            yield _sse({"event": "result", "response": response.model_dump()})
        except HTTPException as e:  # e.g. 503 when the LLM provider is busy
            yield _sse({"event": "error", "status": e.status_code, "detail": e.detail,
                        "retry_after": (e.headers or {}).get("Retry-After")})
        except Exception:
            log.exception("stream failed")
            yield _sse({"event": "error", "status": 500, "detail": "Something went wrong. Please try again."})

    # X-Accel-Buffering: tells a proxy in front of us (nginx) to pass each event through at once
    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


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


@app.post("/analyze/stream")
@limiter.limit(RATE_LIMIT)
def analyze_stream(request: Request, response: Response, body: AnalyzeRequest):
    session_id = str(body.session_id or uuid.uuid4())
    return _stream({"messages": [HumanMessage(body.query)]}, session_id)


@app.post("/resume/stream")
@limiter.limit(RATE_LIMIT)
def resume_stream(request: Request, response: Response, body: ResumeRequest):
    session_id = str(body.session_id)
    state = agent_graph.get_state({"configurable": {"thread_id": session_id}})
    if not (state.tasks and state.tasks[0].interrupts):
        raise HTTPException(status_code=409, detail="This session is not waiting for an answer.")
    return _stream(Command(resume=body.answer), session_id)


# ---------- market strip for the page (no LLM, no tokens) ----------

MARKET_SYMBOLS = ["BTC-USD", "ETH-USD", "GC=F"]
MARKETS_TTL_SECONDS = 60
_markets = {"at": 0.0, "data": None}
_markets_lock = threading.Lock()


class Quote(BaseModel):
    symbol: str
    last: float | None = None
    change_pct: float | None = None   # vs 24 hourly bars ago
    spark: list[float] = []           # last 24 hourly closes
    error: str | None = None


class NextEvent(BaseModel):
    event: str
    hours_away: float


class Markets(BaseModel):
    assets: list[Quote]
    next_event: NextEvent | None = None


def _quote(symbol: str) -> Quote:
    try:
        closes = yf.Ticker(symbol).history(period="5d", interval="1h")["Close"].dropna()
        if len(closes) < 2:
            return Quote(symbol=symbol, error="no recent prices")
        last, day_ago = closes.iloc[-1], closes.iloc[-25] if len(closes) >= 25 else closes.iloc[0]
        return Quote(symbol=symbol, last=round(float(last), 2), change_pct=round(float((last / day_ago - 1) * 100), 2),
                     spark=[round(float(c), 2) for c in closes.iloc[-24:]])
    except Exception:
        log.exception("quote failed for %s", symbol)
        return Quote(symbol=symbol, error="price unavailable")


def _next_event() -> NextEvent | None:
    calendar = get_economic_calendar.invoke({"hours_ahead": 48})
    high = [e for e in calendar.get("events", []) if e["impact"] == "High"]
    return NextEvent(event=high[0]["event"], hours_away=high[0]["hours_away"]) if high else None


@app.get("/markets", response_model=Markets)
@limiter.limit("30/minute")
def markets(request: Request, response: Response):
    with _markets_lock:  # one refresh at a time; everyone else gets the cached copy
        if _markets["data"] is None or time.time() - _markets["at"] > MARKETS_TTL_SECONDS:
            _markets.update(at=time.time(), data=Markets(assets=[_quote(s) for s in MARKET_SYMBOLS], next_event=_next_event()))
        return _markets["data"]


@app.get("/health")
def health():
    return {"status": "ok", "predictions_storage": db.enabled()}


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Log the details server-side; never send stack traces or internals to the client."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})
