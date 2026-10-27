"""Stage 1 dev script: run tools by hand, no LLM involved.

Usage:
    python try_tools.py
"""

import json

from dotenv import load_dotenv

load_dotenv("../.env")  # docker-compose does this for the container; scripts must do it themselves

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from tools import (  # noqa: E402
    _google_news,
    get_crypto_fear_greed,
    get_economic_calendar,
    get_funding_rate,
    get_macro_snapshot,
    get_price_action,
    get_social_sentiment,
    search_news,
)

TOOLS = [
    get_price_action,
    get_macro_snapshot,
    get_economic_calendar,
    get_crypto_fear_greed,
    get_funding_rate,
    search_news,
    get_social_sentiment,
]

# 1. What bind_tools will send the LLM (built from each docstring + type hints).
print("=== Menu card the LLM sees ===")
for t in TOOLS:
    print(json.dumps(convert_to_openai_tool(t), indent=2))

# 2. Real calls. This is exactly what tool_executor will do with a tool_call's args.
for t, args in [
    (get_price_action, {"symbol": "BTC-USD", "timeframe": "4H"}),
    (get_price_action, {"symbol": "GC=F", "timeframe": "1D"}),
    (get_macro_snapshot, {}),
    (get_economic_calendar, {"hours_ahead": 72}),
    (get_crypto_fear_greed, {}),
    (get_funding_rate, {"symbol": "BTC-USD"}),
    (get_funding_rate, {"symbol": "ETH-USD"}),
    (search_news, {"query": "gold price", "time_range": "day"}),  # uses 1 Tavily credit
    (get_social_sentiment, {"symbol": "BTC-USD"}),
    (get_social_sentiment, {"symbol": "GC=F"}),
]:
    print(f"\n=== {t.name}({args}) ===")
    print(json.dumps(t.invoke(args), indent=2))

# The news backup source, called directly (normally only used when Tavily fails).
print("\n=== backup news source: _google_news('gold price', 'day') ===")
print(json.dumps(_google_news("gold price", "day"), indent=2))

# 3. Deliberately bad args, to prove the guard works. An error here is the GOOD outcome.
#    In the real agent, tool_executor (Stage 2) catches it and tells the LLM instead of crashing.
for t, args in [
    (get_price_action, {"symbol": "DOGE-USD", "timeframe": "4H"}),
    (get_economic_calendar, {"hours_ahead": 500}),
    (get_funding_rate, {"symbol": "GC=F"}),  # gold is not on this tool's menu card
    (search_news, {"query": "x" * 300}),  # over the 100-char limit
]:
    print(f"\n=== {t.name}({str(args)[:60]}) -- should be rejected ===")
    try:
        print(f"NOT rejected (guard is broken!): {t.invoke(args)}")
    except Exception as e:
        print(f"Rejected as expected -> {type(e).__name__}: {str(e).splitlines()[0]}")
