# LangGraph Advanced: ReAct Agent with Cycles & Tool Calling
# Use Case: AI DevOps Assistant with Iterative Problem Solving

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. DevOps Tools ---
@tool
def check_service_health(service_name: str) -> str:
    """Check the health status of a microservice."""
    health_data = {
        "api-gateway": {"status": "degraded", "latency_ms": 850, "error_rate": "12%"},
        "auth-service": {"status": "healthy", "latency_ms": 45, "error_rate": "0.1%"},
        "payment-service": {"status": "down", "latency_ms": None, "error_rate": "100%"},
        "user-service": {"status": "healthy", "latency_ms": 120, "error_rate": "0.5%"},
    }
    data = health_data.get(service_name.lower(), {"status": "unknown"})
    return str(data)

@tool
def get_service_logs(service_name: str, lines: int = 10) -> str:
    """Retrieve recent logs from a service."""
    logs = {
        "payment-service": """ERROR: Connection timeout to database (attempt 3/3)
ERROR: Failed to process payment for order #12847
ERROR: Circuit breaker OPEN - rejecting requests
WARN: Memory usage at 95% - approaching limit
ERROR: OOMKilled - container restarted""",
        "api-gateway": """WARN: High latency detected on /api/v1/payments endpoint
WARN: Upstream payment-service returning 503
INFO: Retry attempt 1/3 for payment-service
WARN: Retry attempt 2/3 for payment-service""",
    }
    return logs.get(service_name.lower(), f"No logs found for {service_name}")

@tool
def scale_service(service_name: str, replicas: int) -> str:
    """Scale a service to the specified number of replicas."""
    return f"✅ Scaled {service_name} to {replicas} replicas. Rollout in progress (ETA: 2 min)"

@tool
def restart_service(service_name: str) -> str:
    """Restart a service to recover from failures."""
    return f"✅ Restart initiated for {service_name}. Service will be back online in ~30 seconds"

@tool
def check_database_connections(service_name: str) -> str:
    """Check database connection pool status for a service."""
    db_status = {
        "payment-service": "Connection pool: 0/20 active, 20/20 failed. DB host: db-prod-01 unreachable",
        "user-service": "Connection pool: 8/20 active, healthy",
    }
    return db_status.get(service_name.lower(), "Database status unknown")

# --- 2. State ---
class DevOpsState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. Agent Node ---
tools = [check_service_health, get_service_logs, scale_service, restart_service, check_database_connections]
llm_with_tools = llm.bind_tools(tools)

def agent(state: DevOpsState) -> dict:
    system_prompt = """You are an expert DevOps AI assistant. When investigating issues:
1. Check service health first
2. Examine logs for root cause
3. Check dependencies (databases, upstream services)
4. Take corrective action (restart/scale)
5. Verify the fix

Be systematic and thorough."""
    
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- 4. Build ReAct Graph (with cycles) ---
tool_node = ToolNode(tools)

graph = StateGraph(DevOpsState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # Routes to tools or END
graph.add_edge("tools", "agent")  # Cycle back to agent after tool execution

app = graph.compile()

# --- 5. Run DevOps Investigation ---
print("=== AI DevOps Assistant ===\n")
incident = "We're getting customer complaints about payment failures. Please investigate and fix the issue."

result = app.invoke({"messages": [HumanMessage(content=incident)]})

print("Incident:", incident)
print("\n--- Investigation & Resolution ---")
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"\n🤖 Agent: {msg.content}")
    elif isinstance(msg, ToolMessage):
        print(f"\n🔧 Tool [{msg.name}]: {msg.content[:100]}...")

print(f"\nTotal reasoning steps: {len(result['messages'])}")
