"""
LangGraph Mastery Series - Episode 01
File 3: Memory and Checkpointing
----------------------------------
Short-term memory: the state itself.
Long-term memory: MemorySaver (dev) or PostgresSaver (prod).
The agent remembers Rio across multiple messages.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

#State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Node
def call_llm(state: AgentState) -> dict:
    system = SystemMessage(
        "You are a helpful assistant. Remember everything the "
        "user tells you about themselves."
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

# Graph with checkpointer
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# MemorySaver stores state in RAM -- great for development.
# Swap for PostgresSaver for production persistence.
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Simulate a multi-turn conversation
if __name__ == "__main__":
    # thread_id namespaces memory per user / session
    rio_config = {"configurable": {"thread_id": "user-rio"}}
    bob_config   = {"configurable": {"thread_id": "user-bob"}}

    conversations = [
        (rio_config, "My name is Rio and I am a data scientist."),
        (rio_config, "I love working with Python and machine learning."),
        (rio_config, "What do you know about me so far?"),
        (bob_config,   "Hi, my name is Bob."),
        (bob_config,   "What is my name?"),
        # Rio's thread is completely separate from Bob's
        (rio_config, "What is my profession?"),
    ]

    for config, message in conversations:
        user = config["configurable"]["thread_id"]
        print(f"\n[{user}] User : {message}")
        result = app.invoke(
            {"messages": [HumanMessage(message)]},
            config=config
        )
        print(f"[{user}] Agent: {result['messages'][-1].content}")

#Production swap (commented out)
# from langgraph.checkpoint.postgres import PostgresSaver
# saver = PostgresSaver.from_conn_string("postgresql://user:pass@localhost/db")
# saver.setup()  # creates the required tables once
# app = graph.compile(checkpointer=saver)

