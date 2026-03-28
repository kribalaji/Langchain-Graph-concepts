# LlamaIndex Basics 01: Documents, Nodes, Index & Query Engine
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Simple Knowledge Base Q&A

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# ── 1. Model Setup ───────────────────────────────────────────
# LlamaIndex uses Settings as global config (replaces ServiceContext)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm         = Ollama(model="mistral:7b", request_timeout=120.0)

print("=" * 50)
print("1. DOCUMENTS & NODES")
print("=" * 50)

# ── 2. Create Documents ──────────────────────────────────────
# Document = raw text + metadata (equivalent to LangChain Document)
documents = [
    Document(
        text="""LlamaIndex is a data framework for LLM applications.
        It provides tools to ingest, structure, and access private or domain-specific data.
        Key components: data connectors, indexes, query engines, and chat engines.
        LlamaIndex supports over 160 data sources including PDFs, databases, and APIs.""",
        metadata={"source": "llamaindex_overview", "topic": "framework"}
    ),
    Document(
        text="""A VectorStoreIndex in LlamaIndex stores document embeddings for semantic search.
        It splits documents into nodes, embeds each node, and stores them in a vector store.
        At query time, it retrieves the top-k most similar nodes and passes them to the LLM.
        Supported vector stores: FAISS, Chroma, Pinecone, Weaviate, and more.""",
        metadata={"source": "index_docs", "topic": "indexing"}
    ),
    Document(
        text="""LlamaIndex Query Engine processes natural language questions over indexed data.
        It retrieves relevant nodes, synthesizes an answer using the LLM, and returns a Response.
        The Response object contains the answer text and source nodes with metadata.
        Query engines support streaming, async, and structured output modes.""",
        metadata={"source": "query_docs", "topic": "querying"}
    ),
]

print(f"Created {len(documents)} documents")
for doc in documents:
    print(f"  - [{doc.metadata['topic']}] {doc.text[:60]}...")

# ── 3. Node Parsing ──────────────────────────────────────────
print("\n" + "=" * 50)
print("2. NODE PARSING (chunking)")
print("=" * 50)

# Nodes = chunks of documents (equivalent to LangChain text splitter output)
parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
nodes  = parser.get_nodes_from_documents(documents)

print(f"Total nodes: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"\n  Node {i+1}:")
    print(f"    Text    : {node.text[:80]}...")
    print(f"    Metadata: {node.metadata}")

# ── 4. Build VectorStoreIndex ────────────────────────────────
print("\n" + "=" * 50)
print("3. BUILD VECTOR INDEX (Ollama embeddings)")
print("=" * 50)

index = VectorStoreIndex.from_documents(documents, show_progress=True)
print("Index built successfully!")

# ── 5. Query Engine with Ollama ──────────────────────────────
print("\n" + "=" * 50)
print("4. QUERY ENGINE — OLLAMA")
print("=" * 50)

query_engine = index.as_query_engine(similarity_top_k=2)

questions = [
    "What is LlamaIndex?",
    "How does VectorStoreIndex work?",
    "What does a Query Engine return?",
]

for q in questions:
    response = query_engine.query(q)
    print(f"\nQ: {q}")
    print(f"A: {response.response}")
    print(f"Sources: {[n.metadata.get('topic') for n in response.source_nodes]}")

# ── 6. Query Engine with Groq ────────────────────────────────
print("\n" + "=" * 50)
print("5. QUERY ENGINE — GROQ")
print("=" * 50)

groq_llm    = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
groq_engine = index.as_query_engine(llm=groq_llm, similarity_top_k=2)

response = groq_engine.query("What data sources does LlamaIndex support?")
print(f"Q: What data sources does LlamaIndex support?")
print(f"A: {response.response}")
