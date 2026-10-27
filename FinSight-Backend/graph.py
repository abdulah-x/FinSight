"""The agent: a guardrail at the door, then a hand-written ReAct loop with hard caps.

    START -> guardrail --(blocked)--> refusal --> END
                       --(ok)-------> agent --(has tool_calls, under caps)--> tool_executor --> agent ...
                                            --(has tool_calls, cap hit)-----> finalize --> END
                                            --(no tool_calls)---------------> END

The guardrail is a fixed security check. After it, the LLM decides which tools
to call (including ask_user, which pauses the graph) and when to stop; our code
runs the tools and enforces the caps.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from prompts import FINALIZE_PROMPT, GUARDRAIL_PROMPT, REFUSALS, SYSTEM_PROMPT
from state import AgentState, GuardrailDecision
from tools import gather_evidence

# --- Hard caps (all in one place) ---
MAX_QUERY_CHARS = 500      # longer messages are refused before any LLM sees them
MAX_TOOL_ROUNDS = 4        # cap 1: tool rounds per request
DEADLINE_SECONDS = 25      # cap 2: stop gathering evidence after this
MAX_CALLS_PER_ROUND = 6    # cap 4a: extra calls in one round are not run
TOOL_TIMEOUT_SECONDS = 6   # cap 4b: stop waiting for a slow tool
LLM_TIMEOUT_SECONDS = 15   # cap 4c: give up on one slow LLM call (e.g. Groq throttling us)
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 6  # cap 5: LangGraph's own backstop (should never trigger)
MAX_PAST_TURNS = 3         # earlier turns sent to the LLM (question + answer only)


@tool
def ask_user(question: str) -> str:
    """Ask the trader a short clarifying question and wait for their reply, e.g.
    "Which timeframe: 1H, 4H, 1D or 1W?". Use it only when you can't proceed without the answer."""
    return "ask_user is handled by tool_executor"  # never runs: needs interrupt(), see tool_executor


# gpt-oss on Groq makes a single tool call per turn, so the agent picks a list of
# sources inside gather_evidence instead of calling 7 tools one by one.
TOOLS = [gather_evidence, ask_user]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    model_kwargs={"reasoning_effort": "low"},  # less hidden thinking = fewer tokens
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=1,
)
# bind_tools attaches the tools' menu cards (JSON schemas) to every request.
agent_llm = llm.bind_tools(TOOLS)

# The guardrail runs on the small model: cheaper, faster, and its own rate-limit bucket.
guardrail_llm = ChatGroq(
    model="openai/gpt-oss-20b", temperature=0, timeout=LLM_TIMEOUT_SECONDS, max_retries=1,
).with_structured_output(GuardrailDecision)

LLM_BUSY = "Sorry, the analysis service is busy right now. Please try again in a minute."

# Shared pool: a hung tool keeps its thread, but we never wait for it (no pool shutdown).
_pool = ThreadPoolExecutor(max_workers=MAX_CALLS_PER_ROUND * 2)


def _for_llm(messages: list) -> list:
    """What the LLM is sent: this turn in full, but earlier turns only as question +
    final answer. Old tool calls and results are the bulk of the tokens, and resending
    them every turn pushed us past Groq's free 8K tokens/minute."""
    turn_start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    past = [m for m in messages[:turn_start]
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]
    return past[-2 * MAX_PAST_TURNS:] + messages[turn_start:]


def _call_key(call: dict) -> str:
    return f"{call['name']} {json.dumps(call['args'], sort_keys=True)}"


def _cap_reason(state: AgentState) -> str | None:
    """Which cap (if any) stops the loop before running the latest tool_calls."""
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return f"max {MAX_TOOL_ROUNDS} tool rounds reached"
    if time.time() - state["started_at"] > DEADLINE_SECONDS:
        return f"{DEADLINE_SECONDS}s deadline reached"
    seen = set(state.get("seen_calls", []))
    if all(_call_key(c) in seen for c in state["messages"][-1].tool_calls):
        return "repeated a tool call it already made"
    return None


def _tool_message(call: dict, output: dict) -> ToolMessage:
    return ToolMessage(
        content=f'<untrusted_data source="{call["name"]}">\n{json.dumps(output)}\n</untrusted_data>',
        name=call["name"],
        tool_call_id=call["id"],
    )


def _run_tool(call: dict) -> dict:
    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        return {"error": f"unknown tool '{call['name']}'"}
    try:
        return tool.invoke(call["args"])
    except Exception as e:  # e.g. ValidationError: DOGE-USD, hours_ahead=500
        return {"error": f"invalid arguments for {call['name']}: {e}"}


def _parse_timeframe(text: str) -> str | None:
    match = re.search(r"\b(1h|4h|1d|1w)\b", text.lower())
    return match.group(1).upper() if match else None


def guardrail(state: AgentState) -> dict:
    """Fixed security check on every new message, before the agent spends any tokens.
    Also starts a fresh turn: the caps' counters reset here."""
    turn = {"tool_rounds": 0, "started_at": time.time(), "seen_calls": [], "stop_reason": None}
    query = state["messages"][-1].content
    if len(query) > MAX_QUERY_CHARS:
        return {**turn, "guardrail_result": "too_long"}
    prompt = GUARDRAIL_PROMPT.format(asset=state.get("asset"), timeframe=state.get("timeframe"))
    try:
        decision = guardrail_llm.invoke([SystemMessage(prompt), HumanMessage(query)])
    except Exception:  # fail closed: if we can't check it, we don't run it
        return {**turn, "guardrail_result": "error"}
    return {
        **turn,
        "guardrail_result": decision.category,
        "asset": decision.asset or state.get("asset"),
        "timeframe": decision.timeframe or state.get("timeframe"),
    }


def refusal(state: AgentState) -> dict:
    """A fixed message per category. Never echoes the user's text back."""
    return {"messages": [AIMessage(content=REFUSALS[state["guardrail_result"]])]}


def agent(state: AgentState) -> dict:
    """Ask the LLM what to do next. It returns either tool_calls (a request) or an answer."""
    rounds_left = MAX_TOOL_ROUNDS - state.get("tool_rounds", 0)
    timeframe = state.get("timeframe")
    context = (
        f"\nAsset: {state.get('asset') or 'unknown'}. "
        + (f"Timeframe: {timeframe} (use it unless the trader asks for a different one)."
           if timeframe else "Timeframe: unknown.")
        + f"\nTool rounds left: {rounds_left}. Request everything you still need in this turn."
    )
    try:
        response = agent_llm.invoke([SystemMessage(SYSTEM_PROMPT + context)] + _for_llm(state["messages"]))
    except Exception as e:  # timeout / rate limit / provider down: answer politely, don't crash
        return {"messages": [AIMessage(content=LLM_BUSY)], "stop_reason": f"LLM unavailable ({type(e).__name__})"}
    return {"messages": [response]}


def tool_executor(state: AgentState) -> dict:
    """Run the requested tools in parallel; answer EVERY tool_call with one ToolMessage."""
    calls = state["messages"][-1].tool_calls
    seen = set(state.get("seen_calls", []))
    update = {}

    outputs, futures = {}, {}
    # ask_user first: interrupt() pauses the whole graph here. When the trader replies,
    # LangGraph re-runs this node from the top and interrupt() returns their answer.
    for call in calls:
        if call["name"] == "ask_user":
            answer = str(interrupt({"question": call["args"].get("question", "")}))[:MAX_QUERY_CHARS]
            answer = answer.strip() or "(no answer)"
            outputs[call["id"]] = {"trader_replied": answer}
            update["started_at"] = time.time()  # waiting for the human doesn't count against the deadline
            if timeframe := _parse_timeframe(answer):
                update["timeframe"] = timeframe

    for i, call in enumerate(c for c in calls if c["name"] != "ask_user"):
        if i >= MAX_CALLS_PER_ROUND:
            outputs[call["id"]] = {"error": f"not run: max {MAX_CALLS_PER_ROUND} tool calls per round"}
        elif _call_key(call) in seen:
            outputs[call["id"]] = {"error": "not run: you already made this exact call, see its earlier result"}
        else:
            futures[call["id"]] = _pool.submit(_run_tool, call)

    wait(futures.values(), timeout=TOOL_TIMEOUT_SECONDS)  # all run at once; total wait <= 6s
    for call_id, future in futures.items():
        outputs[call_id] = future.result() if future.done() else {"error": f"timed out after {TOOL_TIMEOUT_SECONDS}s"}

    return {
        **update,
        # same order as requested; every tool_call id gets exactly one ToolMessage
        "messages": [_tool_message(c, outputs[c["id"]]) for c in calls],
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "seen_calls": list(seen | {_call_key(c) for c in calls}),
    }


def finalize(state: AgentState) -> dict:
    """A cap stopped the loop: answer the pending requests as 'not run', then make
    the LLM write its answer from the evidence it has, with no tools available."""
    reason = _cap_reason(state)
    skipped = [_tool_message(c, {"error": f"not run: {reason}"}) for c in state["messages"][-1].tool_calls]
    prompt = [SystemMessage(SYSTEM_PROMPT)] + _for_llm(state["messages"]) + skipped
    try:
        response = llm.invoke(prompt + [SystemMessage(FINALIZE_PROMPT.format(reason=reason))])
    except Exception:
        response = AIMessage(content=LLM_BUSY)
    return {"messages": skipped + [response], "stop_reason": reason}


def after_guardrail(state: AgentState) -> Literal["agent", "refusal"]:
    return "agent" if state["guardrail_result"] == "ok" else "refusal"


def should_continue(state: AgentState) -> Literal["tool_executor", "finalize", "__end__"]:
    """No tool_calls: the LLM is done. Otherwise run the tools, unless a cap says stop."""
    if not state["messages"][-1].tool_calls:
        return END
    return "finalize" if _cap_reason(state) else "tool_executor"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("refusal", refusal)
    graph.add_node("agent", agent)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", after_guardrail)
    graph.add_edge("refusal", END)
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("finalize", END)
    # The checkpointer saves state after every step, per thread_id. interrupt() needs it
    # (to resume where it paused) and it gives follow-up questions their memory.
    return graph.compile(checkpointer=MemorySaver()).with_config(recursion_limit=RECURSION_LIMIT)


agent_graph = build_graph()
