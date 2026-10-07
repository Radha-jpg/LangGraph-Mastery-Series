"""
LangGraph Mastery Series - Episode 01
File 1: First LangGraph App
-------------------------------
A minimal StateGraph with one node.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

#1. Define the state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

#2. Create the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define the node
def call_llm(state: AgentState) -> dict:
    """A node is just a Python function.
    It receives the current state and returns a dict of updates."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

#4. Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

#5. Compile and run
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage("What is LangGraph in one sentence?")]
    })
    print("\nAnswer:", result["messages"][-1].content)
