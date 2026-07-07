# FinSight

A small LangChain + LangGraph financial research agent. See [CLAUDE.md](./CLAUDE.md) for full
project details (architecture, tools, rationale).

## Run it

```bash
cp .env.example .env
# fill in GROQ_API_KEY and TAVILY_API_KEY in .env
docker compose up --build
```

Open http://localhost:8080 and enter a company name or ticker (e.g. "Apple" or "AAPL").
