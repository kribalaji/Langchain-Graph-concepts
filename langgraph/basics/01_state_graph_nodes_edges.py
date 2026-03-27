# LangGraph Basics: State Graphs, Nodes & Edges
# Use Case: Structured Content Generation Workflow

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- 1. Define State Schema ---
class BlogState(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str
    word_count: int

# --- 2. Define Nodes (each is a pure function) ---
def generate_outline(state: BlogState) -> BlogState:
    response = llm.invoke(f"Create a 3-point outline for a blog post about: {state['topic']}")
    return {"outline": response.content}

def write_draft(state: BlogState) -> BlogState:
    response = llm.invoke(
        f"Write a short blog post based on this outline:\n{state['outline']}\n\nTopic: {state['topic']}"
    )
    return {"draft": response.content}

def polish_content(state: BlogState) -> BlogState:
    response = llm.invoke(
        f"Polish and improve this blog post. Make it more engaging:\n\n{state['draft']}"
    )
    final = response.content
    return {"final": final, "word_count": len(final.split())}

def review_quality(state: BlogState) -> BlogState:
    print(f"\n[Quality Check] Word count: {state['word_count']}")
    return state

# --- 3. Build Graph ---
graph = StateGraph(BlogState)

graph.add_node("outline", generate_outline)
graph.add_node("draft", write_draft)
graph.add_node("polish", polish_content)
graph.add_node("review", review_quality)

# --- 4. Define Edges (flow) ---
graph.add_edge(START, "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "polish")
graph.add_edge("polish", "review")
graph.add_edge("review", END)

app = graph.compile()

# --- 5. Run the Graph ---
print("=== Blog Generation Workflow ===")
result = app.invoke({"topic": "Why LangGraph is the future of AI agents"})

print(f"Topic: {result['topic']}")
print(f"\nOutline:\n{result['outline']}")
print(f"\nFinal Post ({result['word_count']} words):\n{result['final'][:500]}...")

# --- 6. Visualize Graph Structure ---
print("\n=== Graph Nodes ===")
for node in graph.nodes:
    print(f"  - {node}")
