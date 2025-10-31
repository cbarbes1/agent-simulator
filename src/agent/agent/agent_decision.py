from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END

# -----------------------------
# 1. Define shared state
# -----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    decision: str
    llm_calls: int

# -----------------------------
# 2. Initialize model
# -----------------------------
model = ChatOllama(model="llama3.2:latest", temperature=0)

# -----------------------------
# 3. Nodes
# -----------------------------
def user_decision(state: MessagesState):
    """Decide next path based on the latest user message."""
    user_msg = state["messages"][-1].content.lower()
    if "assess" in user_msg:
        state["decision"] = "assess_danger"
    elif "action" in user_msg:
        state["decision"] = "perform_action"
    else:
        state["decision"] = "assess_danger"
    return state


def assess_danger(state: MessagesState):
    """AI assesses danger."""
    response = model.invoke([
        SystemMessage(content="You are a soldier assessing danger for your squad."),
        HumanMessage(content="Describe what you observe and estimate the threat level ahead.")
    ])
    if isinstance(response, AIMessage):
        state["messages"].append(response)
    else:
        state["messages"].append(AIMessage(content=str(response)))
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


def perform_action(state: MessagesState):
    """AI performs a tactical action."""
    response = model.invoke([
        SystemMessage(content="You are a squad leader giving orders."),
        HumanMessage(content="Describe in detail how the team carries out the tactical action.")
    ])
    if isinstance(response, AIMessage):
        state["messages"].append(response)
    else:
        state["messages"].append(AIMessage(content=str(response)))
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


# -----------------------------
# 4. Router
# -----------------------------
def route_user_decision(state: MessagesState) -> Literal["assess_danger", "perform_action"]:
    return state["decision"]

# -----------------------------
# 5. Build the graph
# -----------------------------
graph = StateGraph(MessagesState)

graph.add_node("user_decision", user_decision)
graph.add_node("assess_danger", assess_danger)
graph.add_node("perform_action", perform_action)

graph.add_edge(START, "user_decision")
graph.add_conditional_edges("user_decision", route_user_decision)
graph.add_edge("assess_danger", END)
graph.add_edge("perform_action", END)

app = graph.compile()

# -----------------------------
# 6. Run Example
# -----------------------------
initial_messages = [HumanMessage(content="assess danger")]
result = app.invoke({"messages": initial_messages, "llm_calls": 0, "decision": ""})

print("\n--- Conversation ---\n")
for msg in result["messages"]:
    print(f"{msg.type.upper()}: {msg.content}\n")

