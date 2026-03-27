# RAGAS Advanced: LangGraph + RAGAS End-to-End Evaluation
# Use Case: Self-Improving RAG System with Automated Evaluation Loop

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. State ---
class EvalLoopState(TypedDict):
    docs: list[Document]
    questions: list[dict]
    k: int
    prompt_style: str
    scores: dict
    iteration: int
    best_config: dict
    best_score: float
    improvement_suggestions: str

# --- 2. Nodes ---
def build_and_evaluate(state: EvalLoopState) -> dict:
    vectorstore = FAISS.from_documents(state["docs"], embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": state["k"]})
    
    if state["prompt_style"] == "strict":
        sys_msg = "Answer ONLY from context. Say 'I don't know' if not in context.\n\nContext: {context}"
    elif state["prompt_style"] == "detailed":
        sys_msg = "Provide a detailed answer using the context. Include all relevant details.\n\nContext: {context}"
    else:
        sys_msg = "Answer based on context.\n\nContext: {context}"
    
    prompt = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{question}")])
    chain = (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for tc in state["questions"]:
        answer = chain.invoke(tc["question"])
        contexts = [doc.page_content for doc in retriever.invoke(tc["question"])]
        eval_data["question"].append(tc["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(tc["ground_truth"])
    
    dataset = Dataset.from_dict(eval_data)
    result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    
    scores = {
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
        "context_recall": round(result["context_recall"], 3),
    }
    scores["avg"] = round(sum(scores.values()) / 4, 3)
    
    print(f"  Iter {state['iteration']+1}: k={state['k']}, prompt={state['prompt_style']} → avg={scores['avg']}")
    return {"scores": scores, "iteration": state["iteration"] + 1}

def update_best(state: EvalLoopState) -> dict:
    current_avg = state["scores"]["avg"]
    if current_avg > state.get("best_score", 0):
        return {
            "best_score": current_avg,
            "best_config": {"k": state["k"], "prompt_style": state["prompt_style"], "scores": state["scores"]}
        }
    return {}

def generate_suggestions(state: EvalLoopState) -> dict:
    scores = state["scores"]
    issues = []
    if scores["faithfulness"] < 0.8:
        issues.append("Low faithfulness: use stricter prompt to reduce hallucinations")
    if scores["context_recall"] < 0.8:
        issues.append("Low recall: increase k to retrieve more documents")
    if scores["context_precision"] < 0.8:
        issues.append("Low precision: use MMR retrieval to reduce redundant chunks")
    
    suggestions = "; ".join(issues) if issues else "All metrics healthy"
    return {"improvement_suggestions": suggestions}

def try_next_config(state: EvalLoopState) -> dict:
    """Cycle through configurations: k=1→2→3, prompt=standard→strict→detailed"""
    configs = [
        {"k": 1, "prompt_style": "standard"},
        {"k": 2, "prompt_style": "strict"},
        {"k": 3, "prompt_style": "detailed"},
    ]
    next_idx = state["iteration"] % len(configs)
    return configs[next_idx]

def should_continue(state: EvalLoopState) -> str:
    if state["iteration"] >= 3:  # Try 3 configs
        return "finalize"
    if state.get("best_score", 0) >= 0.95:  # Early stop if excellent
        return "finalize"
    return "try_next"

def finalize(state: EvalLoopState) -> dict:
    print(f"\n✅ Best config found: {state['best_config']}")
    return {}

# --- 3. Build Optimization Graph ---
graph = StateGraph(EvalLoopState)
graph.add_node("evaluate", build_and_evaluate)
graph.add_node("update_best", update_best)
graph.add_node("suggest", generate_suggestions)
graph.add_node("next_config", try_next_config)
graph.add_node("finalize", finalize)

graph.add_edge(START, "evaluate")
graph.add_edge("evaluate", "update_best")
graph.add_edge("update_best", "suggest")
graph.add_conditional_edges("suggest", should_continue, {"try_next": "next_config", "finalize": "finalize"})
graph.add_edge("next_config", "evaluate")
graph.add_edge("finalize", END)

app = graph.compile()

# --- 4. Run Self-Optimization Loop ---
docs = [
    Document(page_content="LangChain was founded in 2022 by Harrison Chase. It became one of the fastest growing open-source projects."),
    Document(page_content="LangChain raised $25M Series A in 2023. The framework has over 85,000 GitHub stars."),
    Document(page_content="LangGraph is built on top of LangChain and adds graph-based orchestration for complex agent workflows."),
    Document(page_content="RAGAS was created to evaluate RAG pipelines. It provides reference-free evaluation using LLM-as-judge."),
]

questions = [
    {"question": "Who founded LangChain?", "ground_truth": "LangChain was founded by Harrison Chase in 2022."},
    {"question": "How much funding did LangChain raise?", "ground_truth": "LangChain raised $25M in Series A funding."},
    {"question": "What is LangGraph built on?", "ground_truth": "LangGraph is built on top of LangChain."},
    {"question": "What does RAGAS evaluate?", "ground_truth": "RAGAS evaluates RAG pipelines using LLM-as-judge."},
]

print("=== Self-Optimizing RAG Evaluation Loop ===\n")
result = app.invoke({
    "docs": docs,
    "questions": questions,
    "k": 1,
    "prompt_style": "standard",
    "scores": {},
    "iteration": 0,
    "best_config": {},
    "best_score": 0.0,
    "improvement_suggestions": ""
})

print(f"\n=== Final Results ===")
print(f"Best Config: k={result['best_config']['k']}, prompt={result['best_config']['prompt_style']}")
print(f"Best Avg Score: {result['best_score']}")
print(f"Suggestions: {result['improvement_suggestions']}")
