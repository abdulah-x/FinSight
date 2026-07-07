# FinSight

## What this is
A small demo project built to show fluency with **LangChain** and **LangGraph**. It is
intentionally narrow in scope — not a production app, no auth, no persistence, no test suite
beyond a manual smoke test. It exists to prove the tools can be used correctly, not to be a
polished product.

## Problem statement
Given a company name or stock ticker, produce a structured financial research report:
1. Resolve the free-text query to a ticker symbol (via the LLM).
2. Pull real market data (price, market cap, P/E, 52-week range, sector) via `yfinance`.
3. Search recent news about the company via the Tavily search API.
4. Synthesize both into a markdown report (Summary, Key Metrics, Recent News, Sources).

This is a good LangChain/LangGraph demo because it requires real multi-step orchestration
(tool calls + conditional branching + LLM synthesis) rather than a single prompt-response.

## Architecture
```
resolve_ticker --> fetch_market_data --> [conditional] --> fetch_news --> generate_report --> END
                                              |
                                       (invalid ticker)
                                              |
                                             END
```
- **LangGraph** (`FinSight-Backend/graph.py`): defines `AgentState` and wires the above as an
  explicit `StateGraph` with a conditional edge on ticker validity — this is the core LangGraph
  usage being demonstrated (not just a ReAct loop).
- **LangChain** (`FinSight-Backend/tools.py`): wraps `yfinance` as a `@tool` and uses
  `langchain-tavily`'s `TavilySearch` tool; `ChatGroq` is used for ticker resolution and
  report synthesis.
- **FastAPI** (`FinSight-Backend/main.py`): exposes `POST /research` which invokes the compiled
  graph and returns the report.
- **Frontend** (`FinSight-Frontend/index.html`): a minimal static HTML/JS page (served via
  nginx) with a single input box that calls the backend and renders the report.

## Why these tools
- **LangChain**: standard way to wrap external tools (search, market data) into a form an LLM
  agent can call, and to manage the LLM client itself.
- **LangGraph**: gives explicit control over the multi-step flow (fetch → fetch → synthesize)
  with branching, which is more representative of real agent work than a single chain.
- **Groq (`llama-3.3-70b-versatile`)**: free tier, very fast inference, reliable enough for
  ticker extraction and report writing.
- **Tavily**: purpose-built search API for LLM agents with a free tier and native LangChain
  integration.
- **yfinance**: free, no API key, gives real structured market data — a second tool alongside
  search, which shows multi-tool orchestration in the graph.

## Running it
Two containers, wired with docker-compose:
- `FinSight-Backend`: FastAPI + LangChain + LangGraph agent (port 8000).
- `FinSight-Frontend`: nginx serving the static chat UI (port 8080).

```bash
cp .env.example .env
# fill in GROQ_API_KEY and TAVILY_API_KEY in .env
docker compose up --build
```

Then open http://localhost:8080, enter a company name or ticker (e.g. "Apple" or "AAPL"),
and view the generated report. The frontend calls the backend directly at
`http://localhost:8000` (see `window.FINSIGHT_BACKEND_URL` in `index.html` if this needs to
change for a different environment).

## Effort estimate
Small — roughly a day of work: a few hours for the graph/tools, an hour for the API layer,
an hour for the minimal frontend, remainder for Docker wiring and docs.

## Explicit non-goals
- No authentication, no rate limiting, no persistence/database.
- No automated test suite — verification is a manual end-to-end run.
- Frontend is intentionally minimal; a nicer UI is a possible future iteration, not part of
  this demo's scope.
