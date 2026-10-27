"""Chat with the agent in the terminal and see every step it takes.

Usage:
    python cli.py "BTC bias on 1H"     # then type follow-ups; empty line to quit
    python cli.py                      # start with a prompt
"""

import json
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv("../.env")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.types import Command  # noqa: E402

from graph import agent_graph  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on emoji in news/social text


def show(message, elapsed):
    stamp = f"[{elapsed:5.1f}s]"
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n{stamp} AGENT asks for {len(message.tool_calls)} tool(s):")
        for call in message.tool_calls:
            print(f"          -> {call['name']}({json.dumps(call['args'])})")
    elif isinstance(message, ToolMessage):
        body = json.loads(message.content.split("\n", 1)[1].rsplit("\n", 1)[0])  # strip the untrusted_data tags
        print(f"{stamp} TOOL {message.name}:")
        # gather_evidence returns {source: result}; anything else (e.g. an error) prints as one line
        parts = body.items() if message.name == "gather_evidence" and "error" not in body else [("result", body)]
        for source, result in parts:
            text = json.dumps(result, ensure_ascii=False)
            flag = " [ERROR]" if "error" in result else ""
            print(f"          {source}{flag}: {text[:130]}{'...' if len(text) > 130 else ''}")
    elif isinstance(message, AIMessage):
        print(f"\n{stamp} AGENT answers:\n\n{message.content}")


def run_turn(graph_input, config, stats):
    """Stream one graph run. Returns the question if the agent paused to ask one."""
    for update in agent_graph.stream(graph_input, config=config, stream_mode="updates"):
        for node, output in update.items():
            elapsed = time.time() - stats["start"]
            if node == "__interrupt__":
                return output[0].value["question"]
            if node == "guardrail":
                print(f"[{elapsed:5.1f}s] GUARDRAIL: {output['guardrail_result']} "
                      f"(asset={output.get('asset')}, timeframe={output.get('timeframe')})")
                continue
            if node == "refusal":
                print(f"\n[{elapsed:5.1f}s] REFUSED: {output['messages'][0].content}")
                stats["stop_reason"] = "the guardrail (refused)"
                continue
            if node == "finalize":
                print(f"\n[{elapsed:5.1f}s] CAP HIT: {output['stop_reason']} -> finalize")
            stats.update(output)
            for message in output.get("messages", []):
                if isinstance(message, AIMessage):
                    stats["llm_calls"] += 1
                    stats["tokens"] += (message.usage_metadata or {}).get("total_tokens", 0)
                show(message, elapsed)
    return None


def main():
    # One thread_id = one conversation: the checkpointer keeps its state between turns.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    query = " ".join(sys.argv[1:]) or input("You: ").strip()

    while query:
        print(f"\nQUERY: {query}")
        stats = {"start": time.time(), "llm_calls": 0, "tokens": 0}
        graph_input = {"messages": [HumanMessage(query)]}
        try:
            while question := run_turn(graph_input, config, stats):
                answer = input(f"\n[PAUSED] AGENT asks: {question}\nYou: ").strip()
                # continue exactly where it paused (LangGraph rejects an empty resume value)
                graph_input = Command(resume=answer or "(no answer)")
        except GraphRecursionError:  # cap 5: the backstop; the other caps should stop us first
            print("\nERROR: recursion limit hit, the agent was stopped.")

        print(f"\n--- {time.time() - stats['start']:.1f}s | {stats['llm_calls']} agent LLM calls | "
              f"{stats.get('tool_rounds', 0)} tool rounds | {stats['tokens']} tokens | "
              f"stopped by: {stats.get('stop_reason') or 'the model (it chose to answer)'} ---")
        query = input("\nYou (empty line to quit): ").strip()


if __name__ == "__main__":
    main()
