"""
LangGraph Mastery Series - Episode 01
File 2: ReAct Agent with a Tool-Use Cycle
------------------------------------------
The agent loops: Reason -> Act -> Observe -> Reason ...
until it decides it has enough information to answer.
"""
#
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool

load_dotenv()

# Tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    return (
        f"Search results for '{query}':\n"
        f"LangGraph was released in January 2024 by the LangChain "
        f"team. It is used in production at companies like Elastic, "
        f"Replit, and Uber for building stateful multi-agent systems."
    )

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

tools = [search_web, calculator]

# State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# LLM with tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Agent node
def agent_node(state: AgentState) -> dict:
    # Agent decides what to do next
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Router
def should_continue(state: AgentState) -> str:
    # Decide whether to call a tool or finish
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "end"

# Graph
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Loop back to agent after tool execution
graph.add_edge("tools", "agent")

app = graph.compile()

# Run
if __name__ == "__main__":
    questions = [
        "When was LangGraph released and who made it?",
        "What is 123 multiplied by 456?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")

        result = app.invoke(
            {"messages": [HumanMessage(q)]}
        )

        print(
            f"Answer: {result['messages'][-1].content}"
        )