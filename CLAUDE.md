# FinSight

## What this is
A hand-written ReAct agent that gives a **directional bias** (Bullish / Bearish / Neutral) with a
computed evidence score for **BTC-USD, ETH-USD and GC=F (gold)** on **1H, 4H, 1D and 1W**. Built on
LangGraph without `langgraph.prebuilt`. Runs at $0 on Groq's free tier.

**Hard product rule: bias only, never trade signals.** No entries, stop losses, take profits, sizing
or buy/sell wording, in prompts, output or UI. Key levels appear only as invalidation. Stocks, other
coins and Solana are out of scope (old `SOL-USD` rows remain in the predictions table).

## Architecture
```
guardrail (gpt-oss-20b, structured output)
  ├─ blocked / too long ─> refusal ─> END
  ├─ cache hit ─────────> serve_cached ─> END
  └─ agent (gpt-oss-120b) <─> tool_executor     ReAct loop
        └─ finalize: score_news -> confidence.py -> writer -> END
```
| File | Role |
|---|---|
| `FinSight-Backend/graph.py` | `StateGraph`, nodes, caps, checkpointer selection |
| `FinSight-Backend/prompts.py` | System, guardrail, news-scoring, writer prompts; refusals |
| `FinSight-Backend/tools.py` | Data tools (allowlisted symbols, never raise) and `gather_evidence` |
| `FinSight-Backend/confidence.py` | Pure-Python scoring engine; all thresholds in `CONFIG` |
| `FinSight-Backend/state.py` | `AgentState` and structured-output models |
| `FinSight-Backend/cache.py` | Analysis cache per asset and timeframe |
| `FinSight-Backend/db.py` | Postgres predictions table and conversation pruning |
| `FinSight-Backend/resolver.py` | Grades due predictions |
| `FinSight-Backend/collector.py` | Scheduled forward-test sampling (no agent loop, no writer) |
| `FinSight-Backend/alerts.py` | Discord webhook handler for ERROR logs |
| `FinSight-Backend/main.py` | FastAPI app, SSE streaming, scheduler, `/markets` |
| `FinSight-Backend/eval/` | `golden.json`, `run_eval.py`, `calibration_report.py` |
| `FinSight-Frontend/` | Vanilla JS page (`index.html`, `app.js`, `styles.css`), nginx |

### Agent loop
`gather_evidence` is a single tool that fetches its sources in parallel, because gpt-oss on Groq
makes one tool call per turn. `ask_user` calls `interrupt()` when the timeframe is unknown; the client
resumes on the same `thread_id`. Follow-up turns (`follow_up`) use the model **without tools**, so
"why?" can't trigger a new evidence run.

### Hard caps (`graph.py`)
| Cap | Value |
|---|---|
| Query length | 500 chars, refused before any LLM |
| Tool rounds | 4 |
| Wall-clock deadline for gathering | 25 s |
| Calls per round | 6 |
| Tool timeout | 6 s |
| LLM timeout | 15 s |
| Recursion limit | `2 * MAX_TOOL_ROUNDS + 6` (backstop) |
| Past turns sent to the LLM | 3 |

Repeated tool calls also end the loop. Any cap routes to `finalize` with the evidence so far, which
lowers coverage and therefore the score.

### Scoring (`confidence.py`)
The LLM never decides the bias or score. Each source becomes `Evidence` (direction, strength) by a
written rule. Evidence is averaged per group (price, macro, news, positioning, sentiment), then the
groups are weighted by asset class and timeframe.

- `net` = weighted direction, -1 to +1. Neutral when `|net| < 0.15`.
- `score = 100 * |net| * coverage * event_penalty`. Buckets: Low < 35, Medium 35-65, High > 65.
- Event penalty: 0.7 with a high-impact USD event in the window, 0.85 if the calendar is unavailable.
- Fear & Greed and funding are contrarian, and only at extremes.

All values are starting guesses. Tune them only from forward-test data. The score measures how
strongly the evidence agrees; it is **not** a probability, and the UI must not present it as one.

### Data sources
yfinance (price, DXY, US10Y, VIX, S&P 500), Forex Factory calendar, alternative.me Fear & Greed,
Binance funding, Tavily news with Google News RSS fallback, StockTwits. Binance rejects US IPs, so
**never host in a US region**.

## API
| Endpoint | Purpose |
|---|---|
| `POST /analyze`, `POST /analyze/stream` | Run an analysis (JSON or SSE) |
| `POST /resume`, `POST /resume/stream` | Answer an `ask_user` pause; 409 if nothing is pending |
| `GET /markets` | Quotes and next event; no LLM, cached 60 s, 30/min |
| `GET /health` | Status and which storage is active |

SSE events: `checked`, `planned`, `asking`, `gathered`, `agent_done`, then `result` or `error`. The
client reads them from a POST with `fetch`. The site calls the API at `/api/`; nginx proxies it, so
only the web port is public. nginx overwrites `X-Forwarded-For` and uvicorn trusts it, which keeps
per-client rate limits correct. Groq 429s return a "busy" response.

## Storage
| Data | Where | Lifetime |
|---|---|---|
| Predictions and grades | Postgres `predictions` | Permanent |
| Conversations and paused questions | `PostgresSaver` (`MemorySaver` without `DATABASE_URL`) | Pruned after 7 idle days, 03:30 UTC |
| Analysis cache | In process | 1H 10 min, 4H 30 min, 1D 2 h, 1W 12 h |
| Current page session | Browser `localStorage`, `finsight.session.v1` | Until "New conversation" |

## Scheduled jobs (APScheduler in `main.py`, UTC)
Resolver every 15 min; prune daily 03:30. With `COLLECTOR_ENABLED=true`: 1H at :05, 4H every 4 h at
:10, 1D at 00:15, 1W Monday 00:20. Grace period 300 s with coalescing.

A call is correct if price moved more than 0.25 ATR in the predicted direction, or for Neutral stayed
inside that band.

## Evaluation status
`docker exec finsight-backend python eval/calibration_report.py` on 2026-11-09, 167 graded calls (dropped assets such as Solana are stored but not counted):

| | Hit rate |
|---|---|
| FinSight | 31.1% |
| Follow the EMA trend | 32.3% |
| Always Bullish | 32.3% |
| Coin flip (bull/bear) | 40.7% |
| Medium bucket (n=43) / Low bucket (n=124) / High bucket (n=0) | 37.2% / 29.0% / none |

The engine **does not beat its baselines yet**. The report's "Score meaningful? YES" only compares
bucket order. Medium vs Low is not statistically significant at this sample size (z about 1.0), and
overlapping market conditions make samples correlated. Most samples are 1H. Keep collecting before
tuning weights.

`eval/run_eval.py` checks trajectories on the golden set; the pass baseline is in `eval/baseline.txt`.

## Tests
```bash
cd FinSight-Backend
pytest            # offline: pure logic only, no mocks of external services
pytest -m live    # real APIs and a separate finsight_test database
```
Live tests are paced 65 s apart (`groq_budget`). Outages are simulated with real clients holding
revoked keys. One full live run uses about 60-70K of Groq's 200K daily tokens for gpt-oss-120b.

## Security model
Read-only tools with fixed endpoints and allowlisted symbols, so a successful injection can only
distort an analysis, and the score is computed in code. Third-party text is wrapped in
`<untrusted_data>`. Model and source text reaches the DOM only through `textContent`. nginx sets a CSP
with no `unsafe-inline`. Postgres and the backend port bind to localhost. No user auth, by decision:
the `session_id` UUID is the only conversation key.

## Limits
Groq free tier for gpt-oss-120b: 8,000 tokens/min and 200,000/day. A full analysis is about 4,500+
tokens on the agent model, so back-to-back analyses can hit the per-minute limit.

## Running
```bash
cp .env.example .env    # GROQ_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD
docker compose up --build
```
Site: http://localhost:8080 (`FRONTEND_PORT`). Optional: `LANGFUSE_*` for traces,
`DISCORD_WEBHOOK_URL` for error alerts, `COLLECTOR_ENABLED=true` for forward-test sampling.

Never commit `.env` or `FinSight_Interview_Revision.pdf`. Only public market data goes to free-tier
LLM providers.
