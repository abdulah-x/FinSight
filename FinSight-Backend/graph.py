"""LangGraph state graph for the FinSight financial research agent."""

from typing import TypedDict

from langchain_groq import ChatGroq

from tools import get_stock_data, tavily_search
from langgraph.graph import END, StateGraph

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class AgentState(TypedDict):
    query: str
    ticker: str
    market_data: dict
    news: list
    report: str
    error: str | None


def resolve_ticker(state: AgentState) -> AgentState:
    """Use the LLM to turn a free-text query into a stock ticker symbol."""
    response = llm.invoke(
        f"Extract the most likely stock ticker symbol for this query: '{state['query']}'. "
        "Reply with ONLY the ticker symbol, uppercase, nothing else."
    )
    ticker = response.content.strip().upper()
    return {**state, "ticker": ticker}


def fetch_market_data(state: AgentState) -> AgentState:
    data = get_stock_data.invoke({"ticker": state["ticker"]})
    if "error" in data:
        return {**state, "market_data": {}, "error": data["error"]}
    return {**state, "market_data": data, "error": None}


def fetch_news(state: AgentState) -> AgentState:
    company = state["market_data"].get("name") or state["ticker"]
    results = tavily_search.invoke({"query": f"{company} stock news recent"})
    return {**state, "news": results.get("results", [])}


def generate_report(state: AgentState) -> AgentState:
    news_snippets = "\n".join(
        f"- {item.get('title')}: {item.get('content', '')[:200]} (source: {item.get('url')})"
        for item in state["news"]
    )
    prompt = f"""You are a financial research assistant. Using the data below, write a
concise structured research report in markdown with sections: Summary, Key Metrics,
Recent News, Sources.

Market data:
{state['market_data']}

News:
{news_snippets}
"""
    response = llm.invoke(prompt)
    return {**state, "report": response.content}


def has_valid_ticker(state: AgentState) -> str:
    return "error" if state.get("error") else "ok"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("resolve_ticker", resolve_ticker)
    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("resolve_ticker")
    graph.add_edge("resolve_ticker", "fetch_market_data")
    graph.add_conditional_edges(
        "fetch_market_data",
        has_valid_ticker,
        {"ok": "fetch_news", "error": END},
    )
    graph.add_edge("fetch_news", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


research_graph = build_graph()
