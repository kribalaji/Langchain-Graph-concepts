# LangGraph Intermediate: Human-in-the-Loop
# Use Case: AI Email Draft with Human Approval Before Sending

from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- 1. State ---
class EmailState(TypedDict):
    recipient: str
    subject: str
    context: str
    draft: str
    human_feedback: str
    approved: bool
    send_status: str

# --- 2. Nodes ---
def draft_email(state: EmailState) -> dict:
    prompt = f"""Write a professional email:
To: {state['recipient']}
Subject: {state['subject']}
Context: {state['context']}

Write only the email body."""
    draft = llm.invoke(prompt).content
    print(f"\n📧 Draft Email:\n{'-'*40}\n{draft}\n{'-'*40}")
    return {"draft": draft}

def apply_feedback(state: EmailState) -> dict:
    if not state.get("human_feedback"):
        return {}
    prompt = f"""Revise this email based on feedback:

Original:
{state['draft']}

Feedback: {state['human_feedback']}

Return only the revised email body."""
    revised = llm.invoke(prompt).content
    print(f"\n✏️ Revised Email:\n{'-'*40}\n{revised}\n{'-'*40}")
    return {"draft": revised}

def send_email(state: EmailState) -> dict:
    # Simulated send
    print(f"\n✅ Email sent to {state['recipient']}!")
    return {"send_status": f"Sent to {state['recipient']} at 2024-01-15 10:30 UTC"}

def cancel_email(state: EmailState) -> dict:
    print("\n❌ Email cancelled by user")
    return {"send_status": "Cancelled by user"}

# --- 3. Conditional Edge ---
def route_approval(state: EmailState) -> Literal["send_email", "cancel_email", "apply_feedback"]:
    if state.get("approved") is True:
        return "send_email"
    elif state.get("approved") is False:
        return "cancel_email"
    elif state.get("human_feedback"):
        return "apply_feedback"
    return "cancel_email"

# --- 4. Build Graph with Interrupt ---
checkpointer = MemorySaver()
graph = StateGraph(EmailState)

graph.add_node("draft_email", draft_email)
graph.add_node("apply_feedback", apply_feedback)
graph.add_node("send_email", send_email)
graph.add_node("cancel_email", cancel_email)

graph.add_edge(START, "draft_email")
graph.add_conditional_edges("draft_email", route_approval)
graph.add_conditional_edges("apply_feedback", route_approval)
graph.add_edge("send_email", END)
graph.add_edge("cancel_email", END)

# Interrupt BEFORE approval decision point
app = graph.compile(checkpointer=checkpointer, interrupt_after=["draft_email"])

# --- 5. Workflow Execution ---
config = {"configurable": {"thread_id": "email_001"}}

print("=== Email Approval Workflow ===")
app.invoke({
    "recipient": "client@company.com",
    "subject": "Project Update - Q4 Milestone",
    "context": "Inform client that we completed the Q4 milestone 2 days early and are ready for review",
    "approved": None
}, config=config)

# --- 6. Human Reviews and Approves ---
print("\n👤 Human Review: Approving with minor feedback...")
app.invoke(
    {"human_feedback": "Make it more concise and add a call-to-action", "approved": None},
    config=config
)

# Interrupt again after revision
snapshot = app.get_state(config)
print(f"\nCurrent node: {snapshot.next}")

# Final approval
print("\n👤 Human: Approving final version...")
app.invoke({"approved": True, "human_feedback": ""}, config=config)
