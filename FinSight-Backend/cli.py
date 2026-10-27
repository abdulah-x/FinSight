"""Run the agent from the terminal and print every step it takes.

Usage:
    python cli.py "BTC bias on 1H"
    python cli.py "gold 1W"
"""

import json
import sys
import time

from dotenv import load_dotenv

load_dotenv("../.env")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from graph import agent_graph  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on emoji in news/social text


def show(message, elapsed):
    stamp = f"[{elapsed:5.1f}s]"
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n{stamp} AGENT asks for {len(message.tool_calls)} tool(s):")
        for call in message.tool_calls:
            print(f"          -> {call['name']}({json.dumps(call['args'])})")
    elif isinstance(message, ToolMessage):
        body = message.content.split("\n", 1)[1].rsplit("\n", 1)[0]  # strip the untrusted_data tags
        flag = "  [ERROR]" if '"error"' in body[:12] else ""
        print(f"{stamp} TOOL {message.name}{flag}: {body[:160]}{'...' if len(body) > 160 else ''}")
    elif isinstance(message, AIMessage):
        print(f"\n{stamp} AGENT answers:\n\n{message.content}")


def main():
    query = " ".join(sys.argv[1:]) or "BTC bias on 1H"
    print(f"QUERY: {query}")
    start, tokens, llm_calls = time.time(), 0, 0
    final = {}

    # stream_mode="updates" yields each node's output as soon as that node finishes.
    try:
        for update in agent_graph.stream({"messages": [HumanMessage(query)]}, stream_mode="updates"):
            for node, output in update.items():
                final.update(output)
                if node == "finalize":
                    print(f"\n[{time.time() - start:5.1f}s] CAP HIT: {output['stop_reason']} -> finalize")
                for message in output["messages"]:
                    if isinstance(message, AIMessage):
                        llm_calls += 1
                        tokens += (message.usage_metadata or {}).get("total_tokens", 0)
                    show(message, time.time() - start)
    except GraphRecursionError:  # cap 5: the backstop; the other caps should stop us first
        print("\nERROR: recursion limit hit, the agent was stopped.")

    print(f"\n--- {time.time() - start:.1f}s | {llm_calls} LLM calls | "
          f"{final.get('tool_rounds', 0)} tool rounds | {tokens} tokens | "
          f"stopped by: {final.get('stop_reason', 'the model (it chose to answer)')} ---")


if __name__ == "__main__":
    main()
