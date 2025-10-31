# Step 1: Define tools and model



"""

Design of the basic example agent for the sim so we have a prototype of my thought process

start --> decide next action --> 2 choices:
"""
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import ToolMessage
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage
from typing import Dict, List, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain.messages import HumanMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


model = ChatOllama(
    model="llama3.2:latest",  # or another local model name
    temperature=0
)


# Define tools
@tool
def detect(description: str) -> str:
    """based on the description decide if its safe

    Args: 
        description: str description of the scene
    """
    return "Danger" if "Danger" in description else "Safe"

inventory = {
    "Kill": "Gun",
    "Dig": "Shovel",
    "helo": "Gas",
    "spy": "snoop tech"
}
@tool
def get_best_weapon_for_action(action_type: str, inventory: Dict[str, str]) -> int:
    """Get best weapon for action
    Action types:
        Kill
        Dig
        helo
        spy
    
    Args:
        action_type: str
        inventory: inventory dictionary
    """
    
    return inventory[action_type]


# Augment the LLM with tools
tools = [detect, get_best_weapon_for_action]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: dict):
    """

    LLM decides whether to call a tool or not

    """

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a soldier on the battlefield who must decide if its safe to move forward or depending on the action that needs to be completed you will choose a tool from the inventory"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node(state: dict):
    """
    
    Performs the tool call
    
    """

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}



# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
messages = [HumanMessage(content=f"""Kill the enemy. 
                        Here is the inventory available: 
                        "Kill": "Gun",
                        "Dig": "Shovel",
                        "helo": "Gas",
                        "spy": "snoop tech"
                         """)]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()