from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from tools import Symbol, Timeframe


class GuardrailDecision(BaseModel):
    """What the guardrail model must return (structured output, not free text)."""

    category: Literal["ok", "off_topic", "unsupported_asset", "injection"] = Field(
        description="ok = asks about a supported asset's direction/outlook, or follows up on the conversation"
    )
    asset: Symbol | None = Field(None, description="Asset asked about, if stated or clear from context")
    timeframe: Timeframe | None = Field(None, description="Timeframe, only if stated or clear from context")


class HeadlineScore(BaseModel):
    index: int
    direction: Literal[-1, 0, 1]
    relevance: float = Field(ge=0, le=1)


class NewsScores(BaseModel):
    """What the news-scoring model must return: one score per numbered headline."""

    scores: list[HeadlineScore]


class AgentState(TypedDict):
    # add_messages is a reducer: when a node returns {"messages": [new]}, LangGraph
    # APPENDS `new` to the list instead of replacing it. That's how the conversation
    # grows: Human -> AI(tool_calls) -> Tool, Tool -> AI(tool_calls) -> ... -> AI(answer).
    messages: Annotated[list[AnyMessage], add_messages]
    guardrail_result: str   # the guardrail's category for the latest message
    asset: str | None       # remembered across turns, so "now 4H" still knows the asset
    timeframe: str | None
    tool_rounds: int        # how many times tool_executor has run this turn (cap 1)
    started_at: float       # unix time evidence-gathering started (cap 2: deadline)
    seen_calls: list[str]   # "tool_name {args}" already run this turn (cap 3: no repeats)
    stop_reason: str | None  # why finalize was needed, if a cap stopped the loop
    analysis: dict | None   # this turn's computed bias + score (confidence.Assessment), if any
