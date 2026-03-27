# RAGAS Intermediate: Full RAG Evaluation Pipeline
# Use Case: Evaluate and Compare RAG Configurations (most critical for production)

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

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Knowledge Base ---
docs = [
    Document(page_content="Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity.", metadata={"source": "python.txt"}),
    Document(page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.", metadata={"source": "python.txt"}),
    Document(page_content="Python's package manager pip allows installing thousands of third-party libraries from PyPI.", metadata={"source": "python.txt"}),
    Document(page_content="FastAPI is a modern Python web framework for building APIs. It uses Python type hints and is based on Starlette.", metadata={"source": "fastapi.txt"}),
    Document(page_content="FastAPI automatically generates OpenAPI documentation. It supports async/await and is one of the fastest Python frameworks.", metadata={"source": "fastapi.txt"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)

# --- 2. RAG Pipeline Factory ---
def build_rag_pipeline(k: int = 2, prompt_style: str = "standard"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    if prompt_style == "standard":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context only.\n\nContext: {context}"),
            ("human", "{question}")
        ])
    else:  # strict
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer ONLY using the provided context. 
If the answer is not in the context, say "I don't know."
Context: {context}"""),
            ("human", "{question}")
        ])
    
    return (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    ), retriever

# --- 3. Evaluation Questions ---
questions = [
    "Who created Python?",
    "What programming paradigms does Python support?",
    "What is FastAPI and what is it based on?",
    "Does FastAPI support async programming?",
]

ground_truths = [
    "Python was created by Guido van Rossum and released in 1991.",
    "Python supports procedural, object-oriented, and functional programming.",
    "FastAPI is a modern Python web framework for building APIs, based on Starlette.",
    "Yes, FastAPI supports async/await.",
]

# --- 4. Run Evaluation for Multiple Configs ---
def evaluate_rag(k: int, prompt_style: str) -> dict:
    chain, retriever = build_rag_pipeline(k=k, prompt_style=prompt_style)
    
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    for question, truth in zip(questions, ground_truths):
        answer = chain.invoke(question)
        contexts = [doc.page_content for doc in retriever.invoke(question)]
        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(truth)
    
    dataset = Dataset.from_dict(eval_data)
    result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return result

print("=== RAG Configuration Comparison ===\n")

configs = [
    {"k": 1, "prompt_style": "standard"},
    {"k": 3, "prompt_style": "standard"},
    {"k": 3, "prompt_style": "strict"},
]

comparison = []
for cfg in configs:
    print(f"Evaluating: k={cfg['k']}, prompt={cfg['prompt_style']}...")
    result = evaluate_rag(**cfg)
    scores = {
        "config": f"k={cfg['k']}, {cfg['prompt_style']}",
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
        "context_recall": round(result["context_recall"], 3),
    }
    scores["avg_score"] = round(sum(v for k, v in scores.items() if k != "config") / 4, 3)
    comparison.append(scores)

# --- 5. Results Summary ---
print("\n=== Evaluation Results ===")
print(f"{'Config':<25} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'Avg':>7}")
print("-" * 65)
for s in comparison:
    print(f"{s['config']:<25} {s['faithfulness']:>7} {s['answer_relevancy']:>7} {s['context_precision']:>7} {s['context_recall']:>7} {s['avg_score']:>7}")

best = max(comparison, key=lambda x: x["avg_score"])
print(f"\n✅ Best config: {best['config']} (avg score: {best['avg_score']})")
