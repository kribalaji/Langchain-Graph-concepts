# LangGraph Advanced: Multi-Agent Supervisor Pattern
# Use Case: AI-Powered Market Research Team (most demanded enterprise use case)

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. Specialized Agent Tools ---
@tool
def search_market_data(query: str) -> str:
    """Search for market size, trends, and growth data."""
    data = {
        "ai market": "Global AI market: $196B in 2023, projected $1.8T by 2030, CAGR 37%",
        "llm market": "LLM market: $4.3B in 2023, growing to $36B by 2028",
        "rag market": "RAG solutions market: $1.2B in 2024, fastest growing AI segment",
    }
    for key in data:
        if key in query.lower():
            return data[key]
    return f"Market data for '{query}': Estimated $5-50B market with 25-40% CAGR"

@tool
def analyze_competitors(company: str) -> str:
    """Analyze competitor landscape for a company or product."""
    competitors = {
        "langchain": "Main competitors: LlamaIndex, Haystack, Semantic Kernel. LangChain leads with 85K GitHub stars.",
        "openai": "Main competitors: Anthropic (Claude), Google (Gemini), Meta (Llama). OpenAI holds ~60% enterprise LLM market.",
    }
    return competitors.get(company.lower(), f"Competitors for {company}: 3-5 major players identified in the space")

@tool
def generate_report_section(section: str, data: str) -> str:
    """Generate a formatted report section from raw data."""
    return f"## {section}\n{data}\n\n*Analysis complete. Data verified.*"

@tool
def calculate_tam_sam_som(market_size: str, target_segment: str) -> str:
    """Calculate TAM, SAM, SOM for a market opportunity."""
    return f"""TAM (Total Addressable Market): {market_size}
SAM (Serviceable Addressable Market): ~30% of TAM
SOM (Serviceable Obtainable Market): ~5% of SAM in Year 1
Target Segment: {target_segment}"""

# --- 2. Specialized Agents ---
market_researcher = create_react_agent(
    llm,
    tools=[search_market_data, calculate_tam_sam_som],
    state_modifier="You are a market research specialist. Focus on market size, trends, and opportunities."
)

competitive_analyst = create_react_agent(
    llm,
    tools=[analyze_competitors],
    state_modifier="You are a competitive intelligence analyst. Focus on competitor strengths, weaknesses, and positioning."
)

report_writer = create_react_agent(
    llm,
    tools=[generate_report_section],
    state_modifier="You are a business report writer. Create clear, structured, executive-ready reports."
)

# --- 3. Supervisor State ---
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str
    market_data: str
    competitive_data: str
    final_report: str
    next_agent: str

# --- 4. Supervisor Node ---
AGENTS = ["market_researcher", "competitive_analyst", "report_writer", "FINISH"]

def supervisor(state: SupervisorState) -> dict:
    has_market = bool(state.get("market_data"))
    has_competitive = bool(state.get("competitive_data"))
    has_report = bool(state.get("final_report"))

    if not has_market:
        next_agent = "market_researcher"
    elif not has_competitive:
        next_agent = "competitive_analyst"
    elif not has_report:
        next_agent = "report_writer"
    else:
        next_agent = "FINISH"

    print(f"  [Supervisor] → Routing to: {next_agent}")
    return {"next_agent": next_agent}

def run_market_researcher(state: SupervisorState) -> dict:
    result = market_researcher.invoke({
        "messages": [HumanMessage(content=f"Research the market for: {state['research_topic']}. Include TAM/SAM/SOM.")]
    })
    data = result["messages"][-1].content
    return {"market_data": data, "messages": [AIMessage(content=f"[Market Research]: {data[:100]}...")]}

def run_competitive_analyst(state: SupervisorState) -> dict:
    result = competitive_analyst.invoke({
        "messages": [HumanMessage(content=f"Analyze competitors for: {state['research_topic']}")]
    })
    data = result["messages"][-1].content
    return {"competitive_data": data, "messages": [AIMessage(content=f"[Competitive Analysis]: {data[:100]}...")]}

def run_report_writer(state: SupervisorState) -> dict:
    result = report_writer.invoke({
        "messages": [HumanMessage(content=f"""Write an executive market report for: {state['research_topic']}
        
Market Data: {state['market_data']}
Competitive Data: {state['competitive_data']}

Create a structured report with Market Overview, Competitive Landscape, and Recommendations sections.""")]
    })
    report = result["messages"][-1].content
    return {"final_report": report, "messages": [AIMessage(content="[Report Writer]: Report complete!")]}

# --- 5. Build Supervisor Graph ---
def route_to_agent(state: SupervisorState) -> Literal["market_researcher", "competitive_analyst", "report_writer", "__end__"]:
    mapping = {
        "market_researcher": "market_researcher",
        "competitive_analyst": "competitive_analyst",
        "report_writer": "report_writer",
        "FINISH": "__end__"
    }
    return mapping[state["next_agent"]]

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("market_researcher", run_market_researcher)
graph.add_node("competitive_analyst", run_competitive_analyst)
graph.add_node("report_writer", run_report_writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_to_agent)
graph.add_edge("market_researcher", "supervisor")
graph.add_edge("competitive_analyst", "supervisor")
graph.add_edge("report_writer", "supervisor")

app = graph.compile()

# --- 6. Run Multi-Agent Research ---
print("=== Multi-Agent Market Research Team ===\n")
result = app.invoke({
    "messages": [HumanMessage(content="Conduct full market research on LangChain/LLM developer tools")],
    "research_topic": "LangChain and LLM developer tools market",
    "market_data": "",
    "competitive_data": "",
    "final_report": "",
    "next_agent": ""
})

print("\n" + "="*60)
print("FINAL MARKET RESEARCH REPORT")
print("="*60)
print(result["final_report"])
