# LangGraph Basics: Message State & Chat Graphs
# Use Case: Stateful Chatbot with Message History

from dotenv import load_dotenv
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- 1. Message State (add_messages handles list merging) ---
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_name: str
    turn_count: int

# --- 2. Nodes ---
def chatbot(state: ChatState) -> dict:
    system = SystemMessage(content=f"You are a helpful assistant. The user's name is {state['user_name']}. Be personable and remember context.")
    all_messages = [system] + state["messages"]
    response = llm.invoke(all_messages)
    return {
        "messages": [response],
        "turn_count": state.get("turn_count", 0) + 1
    }

def log_turn(state: ChatState) -> dict:
    print(f"  [Turn {state['turn_count']}] Messages in history: {len(state['messages'])}")
    return {}

# --- 3. Build Graph ---
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_node("logger", log_turn)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "logger")
graph.add_edge("logger", END)

app = graph.compile()

# --- 4. Multi-turn Conversation ---
print("=== Stateful Chatbot ===")
state = {"messages": [], "user_name": "Alice", "turn_count": 0}

conversations = [
    "Hi! What can you help me with?",
    "I'm learning about LangGraph. Can you explain what it is?",
    "What's my name again?",  # Tests memory
]

for user_input in conversations:
    print(f"\nUser: {user_input}")
    state["messages"].append(HumanMessage(content=user_input))
    result = app.invoke(state)
    state = result
    last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
    print(f"Bot: {last_ai.content[:200]}")

print(f"\nTotal turns: {state['turn_count']}")
print(f"Total messages: {len(state['messages'])}")
