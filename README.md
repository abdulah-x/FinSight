# FinSight

Ask which way Bitcoin, Ethereum or Gold is leaning, and get a clear answer with the evidence behind it.

Type something like "Gold 1D" or "what's the bias on Bitcoin this week?". FinSight goes and checks live prices, the US dollar and interest rates, upcoming economic news, crypto market mood and recent headlines. It then tells you whether the evidence leans **Bullish, Bearish or Neutral**, how strongly it agrees, and what would prove that view wrong.

You can watch it work step by step, and ask follow-up questions like "why?".

**FinSight gives a market view, not trade signals.** It will never tell you where to buy, sell or place a stop. Nothing here is financial advice.

## What it covers

- **Markets:** Bitcoin, Ethereum and Gold
- **Timeframes:** 1 hour, 4 hours, 1 day and 1 week

## How good is it?

FinSight grades its own past calls against what the price actually did, and we publish the result even when it isn't flattering.

As of November 2026, across 187 graded calls, **it was right 28% of the time. That is not yet better than simple rules** like following the recent trend (31%). Most of those calls are short-term, and the sample is still small. Please treat FinSight as a way to see the evidence in one place, not as a prediction you can rely on.

## Try it yourself

You need [Docker](https://www.docker.com/) and two free keys, from [Groq](https://console.groq.com/) and [Tavily](https://tavily.com/).

```bash
git clone https://github.com/abdulah-x/FinSight.git
cd FinSight
cp .env.example .env
# Open .env, add your two keys, and choose a database password
docker compose up --build
```

Then open http://localhost:8080.

It runs on free services, so if lots of questions arrive at once you may be asked to wait a minute.

---

Developers: the full technical details are in [CLAUDE.md](./CLAUDE.md).
