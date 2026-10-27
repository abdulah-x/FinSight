"""The agent loop and its hard caps, tested with a scripted fake LLM (no API key, no network).

The fake LLM returns pre-written AIMessages in order, so we control exactly what
"the model decides" and can check that our loop handles each case correctly.
"""

import time
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

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


@tool
def slow_tool(seconds: float) -> dict:
    """fake: takes `seconds` to answer"""
    time.sleep(seconds)
    return {"slept": seconds}


FAKE_TOOLS = {"get_price_action": get_price_action, "get_crypto_fear_greed": get_crypto_fear_greed,
              "slow_tool": slow_tool}


def run(llm, query="BTC 4H", state=None, **config):
    """Run the graph with `llm` playing both the agent and finalize roles."""
    with patch.object(graph, "agent_llm", llm), patch.object(graph, "llm", llm), \
            patch.dict(graph.TOOLS_BY_NAME, FAKE_TOOLS):
        return graph.agent_graph.invoke({"messages": [HumanMessage(query)], **(state or {})},
                                        config=config or None)


def endless_requests():
    """A model that never stops asking (a different arg each time, so it isn't a repeat)."""
    return [asks(("get_price_action", {"symbol": f"X{i}", "timeframe": "1H"})) for i in range(50)]


# ---------- the basic loop (Stage 2) ----------

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
    assert "stop_reason" not in result  # the model chose to stop; no cap needed

    # each ToolMessage answers its own request, by id, in the order requested
    assert [m.tool_call_id for m in result["messages"][2:4]] == ["call_0", "call_1"]
    # the second LLM call was shown the tool results (the "Observe" step of ReAct)
    assert any(isinstance(m, ToolMessage) and '"trend": "down"' in m.content for m in llm.received[1])


def test_llm_can_stop_without_tools():
    result = run(ScriptedLLM(AIMessage(content="I need an asset and timeframe.")))
    assert len(result["messages"]) == 2
    assert result.get("tool_rounds", 0) == 0


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


def test_agent_is_told_how_many_rounds_are_left():
    llm = ScriptedLLM(asks(("get_crypto_fear_greed", {})), AIMessage(content="done"))
    run(llm)
    assert "Tool rounds left: 4" in llm.received[0][0].content
    assert "Tool rounds left: 3" in llm.received[1][0].content


# ---------- cap 1: max tool rounds ----------

def test_cap1_greedy_llm_is_stopped_after_4_rounds_and_still_answers():
    llm = ScriptedLLM(*endless_requests()[:5], AIMessage(content="Bias: Neutral (partial evidence)"))
    result = run(llm)

    assert result["tool_rounds"] == graph.MAX_TOOL_ROUNDS
    assert result["stop_reason"] == "max 4 tool rounds reached"
    assert result["messages"][-1].content == "Bias: Neutral (partial evidence)"
    # the 5th request was never run, but still got its ToolMessage (API rule)
    assert "not run: max 4 tool rounds" in result["messages"][-2].content
    # finalize told the model why it must answer now
    assert "Stop gathering evidence: max 4 tool rounds reached" in llm.received[-1][-1].content


# ---------- cap 2: deadline ----------

def test_cap2_past_deadline_goes_straight_to_finalize():
    llm = ScriptedLLM(asks(("get_crypto_fear_greed", {})), AIMessage(content="answer from no evidence"))
    result = run(llm, state={"started_at": time.time() - 60})
    assert result["stop_reason"] == "25s deadline reached"
    assert result.get("tool_rounds", 0) == 0  # no tools were run


# ---------- cap 3: repeated call ----------

def test_cap3_repeating_the_same_call_stops_the_loop():
    same = ("get_price_action", {"symbol": "BTC-USD", "timeframe": "4H"})
    llm = ScriptedLLM(asks(same), asks(same), AIMessage(content="done"))
    result = run(llm)
    assert result["tool_rounds"] == 1
    assert result["stop_reason"] == "repeated a tool call it already made"


def test_cap3_a_repeat_mixed_with_new_calls_is_skipped_not_rerun():
    btc = ("get_price_action", {"symbol": "BTC-USD", "timeframe": "4H"})
    llm = ScriptedLLM(asks(btc), asks(btc, ("get_crypto_fear_greed", {})), AIMessage(content="done"))
    messages = run(llm)["messages"]
    second_round = [m for m in messages if isinstance(m, ToolMessage)][1:]
    assert "already made this exact call" in second_round[0].content
    assert '"today": 48' in second_round[1].content


# ---------- cap 4: calls per round, timeouts, parallel ----------

def test_cap4_only_6_calls_run_per_round():
    calls = [("get_price_action", {"symbol": f"S{i}", "timeframe": "1H"}) for i in range(8)]
    llm = ScriptedLLM(asks(*calls), AIMessage(content="done"))
    tool_messages = [m for m in run(llm)["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 8  # every request answered...
    assert sum("not run: max 6 tool calls" in m.content for m in tool_messages) == 2  # ...2 not run


def test_cap4_slow_tool_times_out_without_blocking_the_agent():
    llm = ScriptedLLM(asks(("slow_tool", {"seconds": 3})), AIMessage(content="done"))
    start = time.time()
    with patch.object(graph, "TOOL_TIMEOUT_SECONDS", 0.2):
        result = run(llm)
    assert "timed out" in result["messages"][2].content
    assert time.time() - start < 1.5  # we didn't wait the full 3s


def test_cap4_tools_in_one_round_run_in_parallel():
    calls = [("slow_tool", {"seconds": 0.5}) for _ in range(4)]
    llm = ScriptedLLM(asks(*calls), AIMessage(content="done"))
    start = time.time()
    run(llm)
    assert time.time() - start < 1.2  # 4 x 0.5s one after another would be 2s


# ---------- cap 5: recursion backstop ----------

def test_cap5_recursion_limit_is_set_above_what_the_other_caps_allow():
    # 4 rounds x (agent + tool_executor) + final agent + finalize = 10 steps max
    assert graph.RECURSION_LIMIT > 2 * graph.MAX_TOOL_ROUNDS + 2
    assert graph.agent_graph.config["recursion_limit"] == graph.RECURSION_LIMIT
