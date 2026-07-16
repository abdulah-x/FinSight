# FinSight

Ask about a stock or crypto in plain English — get a real answer back.

Type something like "how's Tesla doing?" or "give me a report on Bitcoin," and FinSight
fetches live price data, recent news, and real investor chatter, then writes you either a
quick one-line answer or a full research report, depending on what you asked for. It also
remembers what you talked about, so you can follow up with "now compare it to Apple" and
it'll know what "it" means.

This is a demo project built to show off **LangChain** and **LangGraph** (two frameworks for
building AI agents) — not a polished, production-ready app. See [CLAUDE.md](./CLAUDE.md) for
the full technical breakdown.

## What it can do

- **Quick answers** — "what's the latest news on Tesla?"
- **Full reports** — "analyze NVDA" → summary, key metrics, news, sentiment, sources
- **Crypto too** — "Ethereum," "Dogecoin," not just stocks
- **Follow-up questions** — it remembers the conversation
- **Politely declines** anything that isn't about a stock or crypto asset

## Tech stack

| Layer | Tech |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) — explicit `StateGraph` with conditional routing + `MemorySaver` session memory |
| LLM tooling | [LangChain](https://github.com/langchain-ai/langchain) — tool wrapping, structured output |
| LLM | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) |
| Market data | [`yfinance`](https://github.com/ranaroussi/yfinance) |
| News search | [Tavily](https://tavily.com/) (`langchain-tavily`) |
| Social sentiment | StockTwits public API |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend | Static HTML/JS chat UI served via nginx |
| Deployment | Docker Compose (backend + frontend containers) |

See [CLAUDE.md](./CLAUDE.md) for the full architecture and design rationale.

## Try it

Requires a free [Groq API key](https://console.groq.com/) and [Tavily API key](https://tavily.com/).

```bash
git clone https://github.com/abdulah-x/FinSight.git
cd FinSight
cp .env.example .env
# add your GROQ_API_KEY and TAVILY_API_KEY to .env
docker compose up --build
```

This starts two containers:
- **backend** (FastAPI + LangChain/LangGraph agent) — `http://localhost:8000`
- **frontend** (chat UI via nginx) — `http://localhost:8080`

Open `http://localhost:8080` and ask about any company or crypto asset (e.g. "Apple", "AAPL", "Bitcoin").
