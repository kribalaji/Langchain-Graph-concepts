# RAGAS Intermediate: Retriever Comparison & Optimization
# Use Case: Find the Best Retrieval Strategy for Production

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
import pandas as pd

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Knowledge Base ---
docs = [
    Document(page_content="AWS Lambda is a serverless compute service that runs code without provisioning servers. It scales automatically.", metadata={"topic": "lambda"}),
    Document(page_content="Lambda functions support Python, Node.js, Java, Go, Ruby, and .NET. Max execution time is 15 minutes.", metadata={"topic": "lambda"}),
    Document(page_content="AWS S3 provides object storage with 99.999999999% durability. It supports versioning, lifecycle policies, and cross-region replication.", metadata={"topic": "s3"}),
    Document(page_content="S3 storage classes include Standard, Intelligent-Tiering, Glacier, and Deep Archive for cost optimization.", metadata={"topic": "s3"}),
    Document(page_content="Amazon RDS supports MySQL, PostgreSQL, Oracle, SQL Server, and MariaDB. It handles backups, patching, and failover automatically.", metadata={"topic": "rds"}),
    Document(page_content="RDS Multi-AZ deployments provide high availability with automatic failover. Read replicas improve read performance.", metadata={"topic": "rds"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)

# --- 2. Retrieval Strategies ---
retrievers = {
    "similarity_k1": vectorstore.as_retriever(search_kwargs={"k": 1}),
    "similarity_k3": vectorstore.as_retriever(search_kwargs={"k": 3}),
    "mmr_k3": vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8}),
    "multi_query": MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm
    ),
}

# --- 3. RAG Chain Builder ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only.\n\nContext: {context}"),
    ("human", "{question}")
])

def build_chain(retriever):
    return (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )

# --- 4. Test Questions ---
test_cases = [
    {"question": "What languages does AWS Lambda support?", "ground_truth": "Lambda supports Python, Node.js, Java, Go, Ruby, and .NET."},
    {"question": "How durable is S3 storage?", "ground_truth": "S3 provides 99.999999999% durability."},
    {"question": "What databases does RDS support?", "ground_truth": "RDS supports MySQL, PostgreSQL, Oracle, SQL Server, and MariaDB."},
    {"question": "How does RDS handle high availability?", "ground_truth": "RDS Multi-AZ deployments provide high availability with automatic failover."},
]

# --- 5. Evaluate All Retrievers ---
results_summary = []

for retriever_name, retriever in retrievers.items():
    print(f"Evaluating: {retriever_name}...")
    chain = build_chain(retriever)
    
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for tc in test_cases:
        answer = chain.invoke(tc["question"])
        contexts = [doc.page_content for doc in retriever.invoke(tc["question"])]
        eval_data["question"].append(tc["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(tc["ground_truth"])
    
    dataset = Dataset.from_dict(eval_data)
    result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    
    results_summary.append({
        "retriever": retriever_name,
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
        "context_recall": round(result["context_recall"], 3),
    })

# --- 6. Comparison Report ---
df = pd.DataFrame(results_summary)
df["avg_score"] = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean(axis=1).round(3)
df = df.sort_values("avg_score", ascending=False)

print("\n=== Retriever Comparison Report ===")
print(df.to_string(index=False))

best = df.iloc[0]
print(f"\n✅ Recommended retriever: {best['retriever']} (avg: {best['avg_score']})")
print(f"   Faithfulness: {best['faithfulness']} | Relevancy: {best['answer_relevancy']}")
print(f"   Precision: {best['context_precision']} | Recall: {best['context_recall']}")
