"""The agent loop, tested with a scripted fake LLM (no API key, no network).

The fake LLM returns pre-written AIMessages in order, so we control exactly what
"the model decides" and can check that our loop handles each case correctly.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

import graph


class ScriptedLLM:
    """Replies with the next scripted message; records what it was sent each time."""

    def __init__(self, *replies):
        self.replies, self.received = list(replies), []

    def invoke(self, messages):
        self.received.append(messages)
        return self.replies.pop(0)


def asks(*calls):
    """An AIMessage requesting tools: asks(("tool_name", {args}), ...)."""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(calls)
    ])


@tool
def get_price_action(symbol: str, timeframe: str) -> dict:
    """fake"""
    return {"symbol": symbol, "trend": "down"}


@tool
def get_crypto_fear_greed() -> dict:
    """fake"""
    return {"today": 48}


FAKE_TOOLS = {"get_price_action": get_price_action, "get_crypto_fear_greed": get_crypto_fear_greed}


def run(llm, query="BTC 4H", **config):
    with patch.object(graph, "agent_llm", llm), patch.dict(graph.TOOLS_BY_NAME, FAKE_TOOLS):
        return graph.agent_graph.invoke({"messages": [HumanMessage(query)]}, config=config or None)


def test_one_round_then_answer():
    llm = ScriptedLLM(
        asks(("get_price_action", {"symbol": "BTC-USD", "timeframe": "4H"}), ("get_crypto_fear_greed", {})),
        AIMessage(content="Bias: Bearish - BTC 4H"),
    )
    result = run(llm)

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage"]
    assert result["tool_rounds"] == 1
    assert result["messages"][-1].content == "Bias: Bearish - BTC 4H"

    # each ToolMessage answers its own request, by id
    assert [m.tool_call_id for m in result["messages"][2:4]] == ["call_0", "call_1"]
    # the second LLM call was shown the tool results (the "Observe" step of ReAct)
    assert any(isinstance(m, ToolMessage) and '"trend": "down"' in m.content for m in llm.received[1])


def test_llm_can_stop_without_tools():
    result = run(ScriptedLLM(AIMessage(content="I need an asset and timeframe.")))
    assert len(result["messages"]) == 2
    assert result.get("tool_rounds", 0) == 0


def test_two_rounds_count_up():
    llm = ScriptedLLM(
        asks(("get_price_action", {"symbol": "GC=F", "timeframe": "1D"})),
        asks(("get_crypto_fear_greed", {})),
        AIMessage(content="done"),
    )
    assert run(llm)["tool_rounds"] == 2


def test_tool_output_is_wrapped_as_untrusted():
    llm = ScriptedLLM(asks(("get_crypto_fear_greed", {})), AIMessage(content="done"))
    tool_message = run(llm)["messages"][2]
    assert tool_message.content.startswith('<untrusted_data source="get_crypto_fear_greed">')
    assert tool_message.content.endswith("</untrusted_data>")


def test_invalid_args_become_an_error_message_not_a_crash():
    # real get_funding_rate: gold isn't on its menu card, so validation fails
    llm = ScriptedLLM(asks(("get_funding_rate", {"symbol": "GC=F"})), AIMessage(content="done"))
    result = run(llm)
    assert "invalid arguments for get_funding_rate" in result["messages"][2].content
    assert result["messages"][-1].content == "done"  # the loop carried on


def test_unknown_tool_becomes_an_error_message():
    llm = ScriptedLLM(asks(("delete_database", {})), AIMessage(content="done"))
    assert "unknown tool 'delete_database'" in run(llm)["messages"][2].content


def test_without_caps_a_greedy_llm_only_stops_at_the_recursion_limit():
    """Why Stage 3 exists: if the model never stops asking, nothing in our loop stops it."""
    endless = ScriptedLLM(*[asks(("get_crypto_fear_greed", {})) for _ in range(50)])
    with pytest.raises(GraphRecursionError):
        run(endless, recursion_limit=10)
    assert len(endless.received) == 5  # 10 steps = 5 agent + 5 tool_executor
