"""
LangGraph Mastery Series - Episode 01
File 4: Human in the Loop
---------------------------
The agent proposes an action, pauses with interrupt(),
waits for human approval, then resumes with Command(resume=...).
START
↓
PLAN
↓
APPROVAL (Pause)
↓
EXECUTE
↓
END
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

#State
class AgentState(TypedDict):
    messages: Annotated[list,add_messages]
    proposed_action:str
    approved: bool

#LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

#Nodes
def plan_node(state: AgentState)-> dict:
    """The agent reads the request and proposes an action."""
    response = llm.invoke(
        state["messages"] + [
            HumanMessage(
                "Based on the user request, propose ONE specific "
                "action you would take. Be concise and specific."
            )
        ]
    )
    return {"proposed_action": response.content}


def approval_node(state:AgentState)-> dict:
    """Pause and ask the human whether to proceed."""
    # interrupt() pauses the graph here and surfaces the
    # payload to whoever called app.invoke() or app.stream()
    decision = interrupt({
        "proposed_action": state["proposed_action"],
        "question": "Do you approve? Type 'approve' or 'reject'."
    })
    approved = decision.strip().lower()=="approve"
    return {"approved": approved}


def execute_node(state: AgentState) -> dict:
    """Only runs if the human approved."""
    if not state["approved"]:
        return {"messages": [HumanMessage(
            "Action was rejected. I will not proceed."
        )]}
    response = llm.invoke(
        state["messages"] + [
            HumanMessage(
                f"The user approved this action: {state['proposed_action']}. "
                f"Confirm it has been done and summarise what happened."
            )
        ]
    )
    return {"messages":[response]}


#Graph
graph = StateGraph(AgentState)
graph.add_node("plan",plan_node)
graph.add_node("approval",approval_node)
graph.add_node("execute",execute_node)

graph.add_edge(START,"plan")
graph.add_edge("plan","approval")
graph.add_edge("approval","execute")
graph.add_edge("execute",END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["approval"])

#Run with human-in-the-loop
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-1"}}
    user_request = "Send an email to the whole team announcing the product launch."

    print("=" * 55)
    print("USER REQUEST:", user_request)
    print("=" * 55)

    # First invoke: runs plan node, then pauses at approval
    result=app.invoke(
        {
            "messages": [HumanMessage(user_request)],
            "proposed_action": "",
            "approved": False,
        },
        config=config
    )

    # The graph is now paused. Show what the agent proposes.
    state=app.get_state(config)
    print("\nAGENT PROPOSES:")
    print(state.values.get("proposed_action", ""))
    print()

    # Ask the human
    human_input = input("Your decision (approve/reject): ").strip()

    # Resume the graph with the human's decision
    final = app.invoke(Command(resume=human_input), config=config)
    print("\nAGENT RESPONSE:")
    print(final["messages"][-1].content)
