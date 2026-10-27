from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages is a reducer: when a node returns {"messages": [new]}, LangGraph
    # APPENDS `new` to the list instead of replacing it. That's how the conversation
    # grows: Human -> AI(tool_calls) -> Tool, Tool -> AI(tool_calls) -> ... -> AI(answer).
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int  # how many times tool_executor has run (the Stage 3 caps use it)
