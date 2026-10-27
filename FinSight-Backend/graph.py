"""The agent: a hand-written ReAct loop.

    START -> agent --(has tool_calls)--> tool_executor --> agent --> ...
                   --(no tool_calls)---> END

The LLM decides which tools to call and when to stop. Our code only runs the
tools and loops. No hard caps yet: those are Stage 3.
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from prompts import SYSTEM_PROMPT
from state import AgentState
from tools import (
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
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    model_kwargs={"reasoning_effort": "low"},  # less hidden thinking = fewer tokens
)
# bind_tools attaches the 7 menu cards (JSON schemas) to every request.
agent_llm = llm.bind_tools(TOOLS)


def agent(state: AgentState) -> dict:
    """Ask the LLM what to do next. It returns either tool_calls (a request) or an answer."""
    response = agent_llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tool_executor(state: AgentState) -> dict:
    """Run every tool the LLM asked for and answer each request with one ToolMessage."""
    last: AIMessage = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            output = {"error": f"unknown tool '{call['name']}'"}
        else:
            try:
                output = tool.invoke(call["args"])
            except Exception as e:  # e.g. ValidationError: DOGE-USD, hours_ahead=500
                output = {"error": f"invalid arguments for {call['name']}: {e}"}
        # Every tool_call id MUST get exactly one ToolMessage, even on error,
        # or the next LLM request is rejected by the API.
        results.append(ToolMessage(
            content=f'<untrusted_data source="{call["name"]}">\n{json.dumps(output)}\n</untrusted_data>',
            name=call["name"],
            tool_call_id=call["id"],
        ))
    return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1}


def should_continue(state: AgentState) -> Literal["tool_executor", "__end__"]:
    """The LLM decides: tool_calls means it wants more evidence; none means it's done."""
    if state["messages"][-1].tool_calls:
        return "tool_executor"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tool_executor", tool_executor)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    return graph.compile()


agent_graph = build_graph()
