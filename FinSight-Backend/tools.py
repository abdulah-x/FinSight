"""LangChain tool wrappers used by the FinSight research graph."""

import os

import yfinance as yf
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(
    max_results=5,
    tavily_api_key=os.environ.get("TAVILY_API_KEY"),
)


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
