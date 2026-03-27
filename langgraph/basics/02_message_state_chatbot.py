# LangGraph Basics 02: Message State & Chatbot
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Stateful Multi-turn Chatbot with Full History

from dotenv import load_dotenv
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

ollama_llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)   # lightweight local
groq_llm   = ChatGroq(model="llama3-8b-8192", temperature=0.3)

# ── 1. Message State ─────────────────────────────────────────
# add_messages reducer: APPENDS new messages instead of overwriting
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_choice: str   # "ollama" or "groq"
    turn_count: int

# ── 2. Chatbot Node ──────────────────────────────────────────
def chatbot(state: ChatState) -> dict:
    system = SystemMessage(content=(
        "You are a helpful AI tutor teaching LangChain and LangGraph. "
        "Be concise, clear, and use examples."
    ))
    all_messages = [system] + state["messages"]

    # Route to chosen model
    llm = groq_llm if state.get("model_choice") == "groq" else ollama_llm
    response = llm.invoke(all_messages)

    return {
        "messages": [response],
        "turn_count": state.get("turn_count", 0) + 1
    }

def logger(state: ChatState) -> dict:
    """Logs turn info without modifying state"""
    model = state.get("model_choice", "ollama")
    print(f"  [Logger] Turn {state['turn_count']} | Model: {model} | History: {len(state['messages'])} msgs")
    return {}

# ── 3. Build Graph ───────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_node("logger",  logger)

graph.add_edge(START,     "chatbot")
graph.add_edge("chatbot", "logger")
graph.add_edge("logger",  END)

app = graph.compile()

# ── 4. Multi-turn Conversation with Ollama ───────────────────
print("=" * 50)
print("CHATBOT WITH OLLAMA (local)")
print("=" * 50)

state = {"messages": [], "model_choice": "ollama", "turn_count": 0}

ollama_questions = [
    "What is LangGraph in simple terms?",
    "How is it different from LangChain?",
    "Can you give me a quick code example?",
]

for q in ollama_questions:
    print(f"\nUser: {q}")
    state["messages"].append(HumanMessage(content=q))
    result = app.invoke(state)
    state = result
    last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
    print(f"Bot : {last_ai.content[:250]}")

# ── 5. Multi-turn Conversation with Groq ────────────────────
print("\n" + "=" * 50)
print("CHATBOT WITH GROQ (cloud)")
print("=" * 50)

state_groq = {"messages": [], "model_choice": "groq", "turn_count": 0}

groq_questions = [
    "What is the add_messages reducer in LangGraph?",
    "Why is it important for chatbots?",
]

for q in groq_questions:
    print(f"\nUser: {q}")
    state_groq["messages"].append(HumanMessage(content=q))
    result = app.invoke(state_groq)
    state_groq = result
    last_ai = [m for m in state_groq["messages"] if isinstance(m, AIMessage)][-1]
    print(f"Bot : {last_ai.content[:250]}")

# ── 6. Inspect Full Message History ─────────────────────────
print("\n" + "=" * 50)
print("FULL MESSAGE HISTORY (Ollama session)")
print("=" * 50)
print(f"Total messages: {len(state['messages'])}")
for i, msg in enumerate(state["messages"]):
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{i+1}] {role}: {msg.content[:80]}...")
