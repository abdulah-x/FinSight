import os

import requests
import yfinance as yf
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(
    max_results=5,
    tavily_api_key=os.environ.get("TAVILY_API_KEY"),
)

def _to_stocktwits_symbol(ticker: str) -> str:
    """Convert a Yahoo Finance ticker (e.g. ETH-USD) to StockTwits format (e.g. ETH.X)."""
    if ticker.endswith("-USD"):
        return f"{ticker[:-4]}.X"
    return ticker


@tool
def get_social_sentiment(ticker: str, limit: int = 5) -> list[dict]:
    """Fetch recent StockTwits messages (real user sentiment/discussion) for a ticker.

    Returns a list of {body, username, sentiment, created_at, url}.
    """
    symbol = _to_stocktwits_symbol(ticker)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    headers = {"User-Agent": "Mozilla/5.0 FinSight/1.0 (financial research demo agent)"}

    try:
        response = requests.get(url, headers=headers, timeout=6)
        response.raise_for_status()
        messages = response.json().get("messages", [])
    except Exception:
        return []

    results = []
    for m in messages[:limit]:
        sentiment_data = m.get("entities", {}).get("sentiment")
        results.append(
            {
                "body": m.get("body"),
                "username": m.get("user", {}).get("username"),
                "sentiment": sentiment_data.get("basic") if sentiment_data else None,
                "created_at": m.get("created_at"),
                "url": f"https://stocktwits.com/symbol/{symbol}",
            }
        )
    return results


@tool
def get_stock_data(ticker: str) -> dict:
    """Fetch current price and key fundamentals for a stock ticker (e.g. AAPL)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"error": f"Failed to fetch market data for ticker '{ticker}': {e}"}

    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        return {"error": f"No market data found for ticker '{ticker}'"}

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
