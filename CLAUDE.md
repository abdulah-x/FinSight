# FinSight

## What this is
A small demo project built to show fluency with **LangChain** and **LangGraph**. It is
intentionally narrow in scope — not a production app, no auth, no persistence, no test suite
beyond a manual smoke test. It exists to prove the tools can be used correctly, not to be a
polished product.

## Problem statement
Given a free-text financial query (a company name, ticker, or crypto asset), the agent:
1. Classifies intent — a quick news/sentiment question vs. a full research report request.
2. Resolves the query to a Yahoo Finance ticker symbol (via the LLM).
3. Pulls real market data (price, market cap, P/E, 52-week range, sector) via `yfinance`.
4. Gathers real-time context from two sources: news via the Tavily search API, and real
   user sentiment/discussion via the StockTwits public API.
5. Synthesizes a response shaped to the classified intent: a 1-3 sentence quick answer, or
   a full markdown report (Summary, Key Metrics, Recent News, Social Sentiment, Sources).
6. Remembers prior turns within a session (via LangGraph's `MemorySaver` checkpointer), so
   follow-up queries like "now compare it to Microsoft" have context from the previous turn.

This is a good LangChain/LangGraph demo because it requires real multi-step orchestration:
tool calls, two layers of conditional branching (ticker validity, then intent), structured
LLM output for classification, and session-scoped memory — not just a single prompt-response
or a linear chain.

## Architecture
```
route_query (classifies intent: quick_news | full_report)
   --> resolve_ticker --> fetch_market_data --> [conditional: invalid ticker -> END]
   --> fetch_sources (Tavily news + StockTwits sentiment, depth scaled by intent)
   --> [conditional on intent]
         quick_news  --> generate_quick_answer --> END
         full_report --> generate_report --> END
```
- **LangGraph** (`FinSight-Backend/graph.py`): defines `AgentState` (including `intent`,
  `social_posts`, and a capped `history` of prior turns) and wires the above as an explicit
  `StateGraph` with two conditional edges — one on ticker validity, one on classified intent.
  The graph is compiled with a `MemorySaver` checkpointer keyed by a per-session `thread_id`,
  so state (including `history`) persists across turns within a session.
- **LangChain** (`FinSight-Backend/tools.py`): wraps `yfinance` and the StockTwits public API
  as `@tool`s, and uses `langchain-tavily`'s `TavilySearch` tool. `ChatGroq` powers ticker
  resolution and report synthesis; `route_query` uses `llm.with_structured_output` with a
  Pydantic `Literal` model for reliable intent classification instead of parsing free-text
  output.
- **FastAPI** (`FinSight-Backend/main.py`): exposes `POST /research`, accepting a
  client-generated `session_id` (falls back to a generated UUID) that's passed as the
  LangGraph `thread_id` so conversational memory works across requests.
- **Frontend** (`FinSight-Frontend/index.html`): a static HTML/JS chat UI (served via nginx)
  that generates a session id per page load, sends it with every request, and renders each
  turn as a chat bubble labeled "Quick Answer" or "Full Report" based on the classified intent.

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
- **StockTwits public API**: free, no auth needed for basic reads, gives real user
  sentiment/discussion for stocks and crypto — a zero-cost substitute for X/Twitter, whose
  API now requires a paid tier for read access. (Reddit's public JSON endpoint was tried
  first but is now blocked by Cloudflare bot detection without OAuth.)

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
