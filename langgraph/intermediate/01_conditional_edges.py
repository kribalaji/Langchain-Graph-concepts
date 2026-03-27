# LangGraph Intermediate: Conditional Edges & Branching
# Use Case: AI Code Review & Auto-Fix Pipeline

from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. State ---
class CodeReviewState(TypedDict):
    code: str
    language: str
    review: str
    has_issues: bool
    fixed_code: str
    test_result: str
    iteration: int

# --- 2. Nodes ---
def analyze_code(state: CodeReviewState) -> dict:
    prompt = f"""Review this {state['language']} code for bugs, security issues, and best practices.
    
Code:
{state['code']}

Respond with:
ISSUES_FOUND: yes/no
REVIEW: <detailed review>"""
    
    response = llm.invoke(prompt).content
    has_issues = "ISSUES_FOUND: yes" in response.lower()
    review = response.split("REVIEW:")[-1].strip() if "REVIEW:" in response else response
    return {"review": review, "has_issues": has_issues}

def fix_code(state: CodeReviewState) -> dict:
    prompt = f"""Fix all issues in this {state['language']} code based on the review.
    
Original Code:
{state['code']}

Review:
{state['review']}

Return ONLY the fixed code, no explanation."""
    
    fixed = llm.invoke(prompt).content
    return {"fixed_code": fixed, "iteration": state.get("iteration", 0) + 1}

def run_tests(state: CodeReviewState) -> dict:
    # Simulated test runner
    code_to_test = state.get("fixed_code") or state["code"]
    has_syntax_error = "def " not in code_to_test and state["language"] == "python"
    result = "FAILED: syntax error" if has_syntax_error else "PASSED: all tests green"
    return {"test_result": result}

def approve_code(state: CodeReviewState) -> dict:
    print(f"\n✅ Code approved after {state.get('iteration', 0)} fix iteration(s)")
    return {}

def reject_code(state: CodeReviewState) -> dict:
    print(f"\n❌ Code rejected: {state['test_result']}")
    return {}

# --- 3. Conditional Edge Functions ---
def should_fix(state: CodeReviewState) -> Literal["fix_code", "run_tests"]:
    return "fix_code" if state["has_issues"] else "run_tests"

def check_tests(state: CodeReviewState) -> Literal["approve_code", "reject_code", "fix_code"]:
    if "PASSED" in state["test_result"]:
        return "approve_code"
    elif state.get("iteration", 0) < 2:  # Max 2 fix attempts
        return "fix_code"
    return "reject_code"

# --- 4. Build Graph ---
graph = StateGraph(CodeReviewState)

graph.add_node("analyze_code", analyze_code)
graph.add_node("fix_code", fix_code)
graph.add_node("run_tests", run_tests)
graph.add_node("approve_code", approve_code)
graph.add_node("reject_code", reject_code)

graph.add_edge(START, "analyze_code")
graph.add_conditional_edges("analyze_code", should_fix)
graph.add_edge("fix_code", "run_tests")
graph.add_conditional_edges("run_tests", check_tests)
graph.add_edge("approve_code", END)
graph.add_edge("reject_code", END)

app = graph.compile()

# --- 5. Test with Buggy Code ---
buggy_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)  # Bug: no zero-division check

def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id  # Bug: SQL injection
    return query
"""

print("=== AI Code Review Pipeline ===")
result = app.invoke({
    "code": buggy_code,
    "language": "python",
    "iteration": 0
})

print(f"\nReview:\n{result['review'][:300]}...")
if result.get("fixed_code"):
    print(f"\nFixed Code:\n{result['fixed_code'][:300]}...")
print(f"\nTest Result: {result['test_result']}")
