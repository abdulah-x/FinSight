"""The agent: a hand-written ReAct loop with hard caps.

    START -> agent --(has tool_calls, under caps)--> tool_executor --> agent --> ...
                   --(has tool_calls, cap hit)-----> finalize --> END
                   --(no tool_calls)---------------> END

The LLM decides which tools to call and when to stop. Our code runs the tools
and enforces the caps, so the loop can never run forever.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from prompts import FINALIZE_PROMPT, SYSTEM_PROMPT
from state import AgentState
from tools import gather_evidence

# --- Hard caps (all in one place) ---
MAX_TOOL_ROUNDS = 4        # cap 1: tool rounds per request
DEADLINE_SECONDS = 25      # cap 2: stop gathering evidence after this
MAX_CALLS_PER_ROUND = 6    # cap 4a: extra calls in one round are not run
TOOL_TIMEOUT_SECONDS = 6   # cap 4b: stop waiting for a slow tool
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 4  # cap 5: LangGraph's own backstop (should never trigger)

# One tool: gpt-oss on Groq makes a single tool call per turn, so the agent picks a
# list of sources inside gather_evidence instead of calling 7 tools one by one.
TOOLS = [gather_evidence]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    model_kwargs={"reasoning_effort": "low"},  # less hidden thinking = fewer tokens
)
# bind_tools attaches the tool's menu card (JSON schema) to every request.
agent_llm = llm.bind_tools(TOOLS)

# Shared pool: a hung tool keeps its thread, but we never wait for it (no pool shutdown).
_pool = ThreadPoolExecutor(max_workers=MAX_CALLS_PER_ROUND * 2)


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


def agent(state: AgentState) -> dict:
    """Ask the LLM what to do next. It returns either tool_calls (a request) or an answer."""
    started_at = state.get("started_at") or time.time()
    rounds_left = MAX_TOOL_ROUNDS - state.get("tool_rounds", 0)
    budget = f"\nTool rounds left: {rounds_left}. Request everything you still need in this turn."
    response = agent_llm.invoke([SystemMessage(SYSTEM_PROMPT + budget)] + state["messages"])
    return {"messages": [response], "started_at": started_at}


def tool_executor(state: AgentState) -> dict:
    """Run the requested tools in parallel; answer EVERY tool_call with one ToolMessage."""
    calls = state["messages"][-1].tool_calls
    seen = set(state.get("seen_calls", []))

    outputs, futures = {}, {}
    for i, call in enumerate(calls):
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
        # same order as requested; every tool_call id gets exactly one ToolMessage
        "messages": [_tool_message(c, outputs[c["id"]]) for c in calls],
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "seen_calls": list(seen | {_call_key(c) for c in calls[:MAX_CALLS_PER_ROUND]}),
    }


def finalize(state: AgentState) -> dict:
    """A cap stopped the loop: answer the pending requests as 'not run', then make
    the LLM write its answer from the evidence it has, with no tools available."""
    reason = _cap_reason(state)
    skipped = [_tool_message(c, {"error": f"not run: {reason}"}) for c in state["messages"][-1].tool_calls]
    prompt = [SystemMessage(SYSTEM_PROMPT)] + state["messages"] + skipped
    response = llm.invoke(prompt + [SystemMessage(FINALIZE_PROMPT.format(reason=reason))])
    return {"messages": skipped + [response], "stop_reason": reason}


def should_continue(state: AgentState) -> Literal["tool_executor", "finalize", "__end__"]:
    """No tool_calls: the LLM is done. Otherwise run the tools, unless a cap says stop."""
    if not state["messages"][-1].tool_calls:
        return END
    return "finalize" if _cap_reason(state) else "tool_executor"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("finalize", END)
    return graph.compile().with_config(recursion_limit=RECURSION_LIMIT)


agent_graph = build_graph()
