# LangGraph Advanced: Parallel Subgraphs & Send API
# Use Case: Concurrent Document Analysis Pipeline

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. States ---
class DocumentState(TypedDict):
    document: str
    doc_id: str
    summary: str
    sentiment: str
    key_topics: list[str]

class AggregatorState(TypedDict):
    documents: list[str]
    analyses: Annotated[list[dict], operator.add]  # Reducer: merges lists
    final_report: str

# --- 2. Per-Document Analysis Nodes ---
def analyze_document(state: DocumentState) -> dict:
    doc = state["document"]
    
    # Run 3 analyses on the document
    summary = llm.invoke(f"Summarize in 1 sentence: {doc}").content
    sentiment = llm.invoke(f"Sentiment (positive/negative/neutral): {doc}. Return one word.").content.strip()
    topics_raw = llm.invoke(f"List 3 key topics as comma-separated words: {doc}").content
    topics = [t.strip() for t in topics_raw.split(",")][:3]
    
    return {
        "summary": summary,
        "sentiment": sentiment,
        "key_topics": topics
    }

def format_analysis(state: DocumentState) -> dict:
    analysis = {
        "doc_id": state["doc_id"],
        "summary": state["summary"],
        "sentiment": state["sentiment"],
        "topics": state["key_topics"]
    }
    print(f"  ✅ Analyzed {state['doc_id']}: {state['sentiment']}")
    return {"analyses": [analysis]}  # Will be merged by reducer

# --- 3. Document Subgraph ---
doc_graph = StateGraph(DocumentState)
doc_graph.add_node("analyze", analyze_document)
doc_graph.add_node("format", format_analysis)
doc_graph.add_edge(START, "analyze")
doc_graph.add_edge("analyze", "format")
doc_graph.add_edge("format", END)
doc_subgraph = doc_graph.compile()

# --- 4. Main Graph Nodes ---
def dispatch_documents(state: AggregatorState):
    """Fan-out: send each document to parallel analysis using Send API"""
    return [
        Send("analyze_doc", {
            "document": doc,
            "doc_id": f"doc_{i+1}",
            "summary": "",
            "sentiment": "",
            "key_topics": []
        })
        for i, doc in enumerate(state["documents"])
    ]

def generate_report(state: AggregatorState) -> dict:
    analyses = state["analyses"]
    
    sentiments = [a["sentiment"] for a in analyses]
    all_topics = [t for a in analyses for t in a["topics"]]
    
    report = f"""=== Document Analysis Report ===
Total Documents: {len(analyses)}
Sentiment Distribution: {dict((s, sentiments.count(s)) for s in set(sentiments))}
Top Topics: {', '.join(set(all_topics)[:5])}

Individual Summaries:
"""
    for a in analyses:
        report += f"\n[{a['doc_id']}] ({a['sentiment']}): {a['summary']}"
    
    return {"final_report": report}

# --- 5. Main Graph with Parallel Execution ---
main_graph = StateGraph(AggregatorState)
main_graph.add_node("dispatch", dispatch_documents)
main_graph.add_node("analyze_doc", doc_subgraph)  # Subgraph as node
main_graph.add_node("report", generate_report)

main_graph.add_conditional_edges("dispatch", lambda s: s, ["analyze_doc"])  # Fan-out via Send
main_graph.add_edge(START, "dispatch")
main_graph.add_edge("analyze_doc", "report")
main_graph.add_edge("report", END)

app = main_graph.compile()

# --- 6. Run Parallel Analysis ---
documents = [
    "LangChain has revolutionized AI application development with its modular framework. Developers love its flexibility.",
    "Some users report LangChain has a steep learning curve and frequent breaking changes between versions.",
    "LangGraph enables complex multi-agent workflows that were previously impossible with simple chains.",
    "RAGAS provides essential evaluation tools that help teams ship reliable RAG systems to production.",
]

print("=== Parallel Document Analysis Pipeline ===\n")
print(f"Processing {len(documents)} documents in parallel...\n")

result = app.invoke({"documents": documents, "analyses": []})

print("\n" + result["final_report"])
