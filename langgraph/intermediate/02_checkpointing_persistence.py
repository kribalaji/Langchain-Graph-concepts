# LangGraph Intermediate: Checkpointing & Persistence
# Use Case: Long-running AI Research Assistant with Resume Capability

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. State with Persistent Fields ---
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str
    findings: list[str]
    sources_checked: int

# --- 2. Nodes ---
def research_node(state: ResearchState) -> dict:
    topic = state.get("research_topic", "general")
    response = llm.invoke(state["messages"] + [
        HumanMessage(content=f"Provide one key finding about: {topic}. Be specific and cite a source.")
    ])
    findings = state.get("findings", [])
    findings.append(response.content)
    return {
        "messages": [response],
        "findings": findings,
        "sources_checked": state.get("sources_checked", 0) + 1
    }

def summarize_node(state: ResearchState) -> dict:
    if len(state.get("findings", [])) < 2:
        return {}
    summary_prompt = f"Summarize these research findings in 2 sentences:\n" + "\n".join(state["findings"])
    response = llm.invoke(summary_prompt)
    return {"messages": [AIMessage(content=f"SUMMARY: {response.content}")]}

# --- 3. Build Graph with Checkpointer ---
checkpointer = MemorySaver()

graph = StateGraph(ResearchState)
graph.add_node("research", research_node)
graph.add_node("summarize", summarize_node)

graph.add_edge(START, "research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", END)

app = graph.compile(checkpointer=checkpointer)

# --- 4. Session-based Persistence ---
thread_id = "research_session_001"
config = {"configurable": {"thread_id": thread_id}}

print("=== Research Session 1 ===")
result1 = app.invoke({
    "messages": [HumanMessage(content="Start researching LangGraph")],
    "research_topic": "LangGraph multi-agent systems",
    "findings": [],
    "sources_checked": 0
}, config=config)

print(f"Findings so far: {len(result1['findings'])}")
print(f"Sources checked: {result1['sources_checked']}")

# --- 5. Resume from Checkpoint ---
print("\n=== Resuming Session (same thread_id) ===")
result2 = app.invoke({
    "messages": [HumanMessage(content="Find another key finding about LangGraph")],
}, config=config)  # Same config = resumes from checkpoint

print(f"Total findings: {len(result2['findings'])}")
print(f"Total sources: {result2['sources_checked']}")
for i, finding in enumerate(result2["findings"], 1):
    print(f"\nFinding {i}: {finding[:150]}...")

# --- 6. Get State Snapshot ---
snapshot = app.get_state(config)
print(f"\n=== Checkpoint Snapshot ===")
print(f"Next nodes: {snapshot.next}")
print(f"Messages in state: {len(snapshot.values.get('messages', []))}")

# --- 7. Different Thread = Fresh Session ---
new_config = {"configurable": {"thread_id": "research_session_002"}}
result3 = app.invoke({
    "messages": [HumanMessage(content="Research RAGAS evaluation framework")],
    "research_topic": "RAGAS metrics",
    "findings": [],
    "sources_checked": 0
}, config=new_config)
print(f"\n=== New Session (thread 002) ===")
print(f"Sources checked: {result3['sources_checked']} (fresh start)")
