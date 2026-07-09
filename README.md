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

## Try it

```bash
cp .env.example .env
# add your GROQ_API_KEY and TAVILY_API_KEY to .env (both are free)
docker compose up --build
```

Then open http://localhost:8080 and ask about any company or crypto asset.
