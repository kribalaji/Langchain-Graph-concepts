# RAGAS Advanced: Custom Metrics & CI/CD Evaluation Pipeline
# Use Case: Automated Quality Gate for RAG Deployments

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from dataclasses import dataclass, field
import pandas as pd
import json
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Custom Metric: Response Conciseness ---
@dataclass
class ResponseConciseness(MetricWithLLM, SingleTurnMetric):
    """Measures if the response is concise without unnecessary padding."""
    name: str = "response_conciseness"
    
    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        prompt = f"""Rate the conciseness of this answer on a scale of 0 to 1.
1.0 = perfectly concise, no fluff
0.5 = some unnecessary content
0.0 = very verbose with lots of padding

Question: {sample.user_input}
Answer: {sample.response}

Return ONLY a number between 0 and 1."""
        response = await self.llm.ainvoke(prompt)
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.5

# --- 2. RAG Pipeline ---
docs = [
    Document(page_content="Python 3.12 was released in October 2023 with performance improvements of 5% on average.", metadata={"source": "python_news.txt"}),
    Document(page_content="Python 3.12 introduces new type parameter syntax for generics using the 'type' keyword.", metadata={"source": "python_news.txt"}),
    Document(page_content="The GIL (Global Interpreter Lock) removal is planned for Python 3.13 as an experimental feature.", metadata={"source": "python_news.txt"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only.\n\nContext: {context}"),
    ("human", "{question}")
])

rag_chain = (
    {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# --- 3. Evaluation Dataset ---
test_cases = [
    {"question": "When was Python 3.12 released?", "ground_truth": "Python 3.12 was released in October 2023."},
    {"question": "What performance improvement does Python 3.12 offer?", "ground_truth": "Python 3.12 offers 5% average performance improvement."},
    {"question": "What is the new syntax for generics in Python 3.12?", "ground_truth": "Python 3.12 introduces the 'type' keyword for generic type parameters."},
    {"question": "What is planned for Python 3.13?", "ground_truth": "GIL removal is planned as an experimental feature in Python 3.13."},
]

eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for tc in test_cases:
    answer = rag_chain.invoke(tc["question"])
    contexts = [doc.page_content for doc in retriever.invoke(tc["question"])]
    eval_data["question"].append(tc["question"])
    eval_data["answer"].append(answer)
    eval_data["contexts"].append(contexts)
    eval_data["ground_truth"].append(tc["ground_truth"])

dataset = Dataset.from_dict(eval_data)

# --- 4. Run Evaluation with Standard Metrics ---
print("=== Running RAGAS Evaluation ===")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# --- 5. CI/CD Quality Gate ---
QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,       # Must be > 85% to prevent hallucinations
    "answer_relevancy": 0.80,   # Must be > 80% relevance
    "context_precision": 0.75,  # Must be > 75% precision
    "context_recall": 0.75,     # Must be > 75% recall
}

def run_quality_gate(results: dict, thresholds: dict) -> dict:
    gate_results = {}
    all_passed = True
    
    for metric, threshold in thresholds.items():
        score = results.get(metric, 0)
        passed = score >= threshold
        gate_results[metric] = {
            "score": round(score, 3),
            "threshold": threshold,
            "passed": passed,
            "gap": round(score - threshold, 3)
        }
        if not passed:
            all_passed = False
    
    return {"passed": all_passed, "metrics": gate_results}

gate = run_quality_gate(results, QUALITY_THRESHOLDS)

print("\n=== CI/CD Quality Gate Results ===")
for metric, data in gate["metrics"].items():
    status = "✅ PASS" if data["passed"] else "❌ FAIL"
    print(f"{status} {metric}: {data['score']} (threshold: {data['threshold']}, gap: {data['gap']:+.3f})")

print(f"\n{'🚀 DEPLOYMENT APPROVED' if gate['passed'] else '🛑 DEPLOYMENT BLOCKED'}")

# --- 6. Evaluation Report ---
report = {
    "timestamp": datetime.now().isoformat(),
    "model": "gpt-4o-mini",
    "num_samples": len(test_cases),
    "gate_passed": gate["passed"],
    "scores": {k: v["score"] for k, v in gate["metrics"].items()},
    "thresholds": QUALITY_THRESHOLDS,
}

with open("evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to evaluation_report.json")

# --- 7. Per-Sample Failure Analysis ---
df = results.to_pandas()
print("\n=== Failure Analysis (scores below threshold) ===")
for metric, threshold in QUALITY_THRESHOLDS.items():
    if metric in df.columns:
        failures = df[df[metric] < threshold][["question", metric]]
        if not failures.empty:
            print(f"\n{metric} failures:")
            for _, row in failures.iterrows():
                print(f"  [{row[metric]:.3f}] {row['question']}")
