from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages is a reducer: when a node returns {"messages": [new]}, LangGraph
    # APPENDS `new` to the list instead of replacing it. That's how the conversation
    # grows: Human -> AI(tool_calls) -> Tool, Tool -> AI(tool_calls) -> ... -> AI(answer).
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int        # how many times tool_executor has run (cap 1)
    started_at: float       # unix time of the first agent step (cap 2: deadline)
    seen_calls: list[str]   # "tool_name {args}" already run (cap 3: no repeats)
    stop_reason: str        # why finalize was needed, if a cap stopped the loop
