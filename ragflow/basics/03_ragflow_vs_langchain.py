# RAGFlow Basics 03: RAGFlow vs LangChain RAG — Side by Side
# Shows the same Q&A task done with both approaches
# Models: Ollama (local) for both

import os
import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:80")
RAGFLOW_API_KEY  = os.environ.get("RAGFLOW_API_KEY", "")

# ── Shared Knowledge ─────────────────────────────────────────
FAQ = [
    ("What is the return policy?",    "Items can be returned within 30 days with receipt."),
    ("How do I reset my password?",   "Go to Settings > Security > Reset Password."),
    ("What are support hours?",       "Support is available 24/7 via chat, Mon-Fri 9-6 EST by phone."),
    ("Does the app work offline?",    "Core features work offline. Sync happens when reconnected."),
    ("How do I upgrade my plan?",     "Go to Account > Billing > Upgrade Plan."),
]

questions = [q for q, _ in FAQ]

# ════════════════════════════════════════════════════════════
print("=" * 60)
print("APPROACH 1: LANGCHAIN RAG (you build the pipeline)")
print("=" * 60)
# ════════════════════════════════════════════════════════════

# Step 1: Create documents
docs = [Document(page_content=f"Q: {q}\nA: {a}") for q, a in FAQ]

# Step 2: Embed + store
embeddings  = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 3: Build RAG chain
llm    = ChatOllama(model="mistral:7b", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only.\n\nContext: {context}"),
    ("human", "{question}")
])
chain = (
    {"context": retriever | (lambda d: "\n".join(x.page_content for x in d)),
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print("\nLangChain RAG Results:")
for q in questions[:3]:
    print(f"\n  Q: {q}")
    print(f"  A: {chain.invoke(q)}")

print("""
LangChain RAG — What YOU manage:
  ✅ Document loading
  ✅ Text splitting strategy
  ✅ Embedding model choice
  ✅ Vector store setup
  ✅ Retriever configuration
  ✅ Prompt engineering
  ✅ LLM selection
""")

# ════════════════════════════════════════════════════════════
print("=" * 60)
print("APPROACH 2: RAGFLOW (managed pipeline via API)")
print("=" * 60)
# ════════════════════════════════════════════════════════════

ragflow_running = False
try:
    r = requests.get(f"{RAGFLOW_BASE_URL}/api/v1/health", timeout=3)
    ragflow_running = r.status_code == 200
except:
    pass

if ragflow_running and RAGFLOW_API_KEY:
    headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}

    # Just call the API — RAGFlow handles everything internally
    assistants = requests.get(f"{RAGFLOW_BASE_URL}/api/v1/assistants", headers=headers).json()
    assistant_list = assistants.get("data", {}).get("assistants", [])

    if assistant_list:
        assistant_id = assistant_list[0]["id"]
        session = requests.post(
            f"{RAGFLOW_BASE_URL}/api/v1/assistants/{assistant_id}/sessions",
            headers=headers, json={"name": "comparison-session"}
        ).json()
        session_id = session.get("data", {}).get("id", "")

        print("\nRAGFlow Results:")
        for q in questions[:3]:
            result = requests.post(
                f"{RAGFLOW_BASE_URL}/api/v1/assistants/{assistant_id}/sessions/{session_id}/chat",
                headers=headers, json={"question": q, "stream": False}
            ).json()
            answer = result.get("data", {}).get("answer", "")
            print(f"\n  Q: {q}")
            print(f"  A: {answer[:150]}")
    else:
        print("No assistants found. Create one in RAGFlow UI first.")
else:
    print("RAGFlow not running. Start with:")
    print("  docker-compose -f ragflow/docker-compose.yml up -d")

print("""
RAGFlow — What IT manages for you:
  ✅ Deep PDF/table/image parsing
  ✅ Smart chunking (QA, table, paper modes)
  ✅ Embedding + vector storage
  ✅ Hybrid search (vector + keyword)
  ✅ Chunk ranking & reranking
  ✅ Citation tracking
  ✅ Web UI for document management
""")

# ── Comparison Summary ───────────────────────────────────────
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"""
  Feature              LangChain RAG     RAGFlow
  ─────────────────────────────────────────────────
  Setup complexity     High              Low (UI)
  PDF/table parsing    Basic             Deep (OCR)
  Chunk strategies     Manual            Auto + 8 modes
  Hybrid search        Manual            Built-in
  Web UI               No                Yes
  API                  Python only       REST API
  Best for             Custom pipelines  Production RAG
  Cost                 Free              Free (self-hosted)
""")
