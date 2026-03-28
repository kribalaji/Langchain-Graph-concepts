# LlamaIndex Basics 03: Retrievers & Response Synthesizers
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Customizable RAG Pipeline Components

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm         = Ollama(model="mistral:7b", request_timeout=120.0)

# ── Knowledge Base ───────────────────────────────────────────
docs = [
    Document(text="AWS EC2 provides resizable compute capacity in the cloud. It supports hundreds of instance types optimized for different workloads.", metadata={"service": "EC2", "category": "compute"}),
    Document(text="AWS Lambda is serverless compute that runs code without managing servers. It scales automatically and charges only for execution time.", metadata={"service": "Lambda", "category": "compute"}),
    Document(text="AWS S3 is object storage with 99.999999999% durability. It supports versioning, lifecycle policies, and static website hosting.", metadata={"service": "S3", "category": "storage"}),
    Document(text="AWS RDS is a managed relational database service supporting MySQL, PostgreSQL, Oracle, SQL Server, and MariaDB.", metadata={"service": "RDS", "category": "database"}),
    Document(text="AWS DynamoDB is a fully managed NoSQL database with single-digit millisecond performance at any scale.", metadata={"service": "DynamoDB", "category": "database"}),
    Document(text="AWS SageMaker is a fully managed ML platform for building, training, and deploying machine learning models at scale.", metadata={"service": "SageMaker", "category": "ml"}),
]

index = VectorStoreIndex.from_documents(docs)
print("Index built!\n")

# ── 1. Default Retriever ─────────────────────────────────────
print("=" * 50)
print("1. VECTOR INDEX RETRIEVER")
print("=" * 50)

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
nodes = retriever.retrieve("serverless compute options")
print(f"Retrieved {len(nodes)} nodes:")
for node in nodes:
    print(f"  Score: {node.score:.4f} | [{node.metadata['service']}] {node.text[:60]}...")

# ── 2. Similarity Postprocessor (filter low scores) ─────────
print("\n" + "=" * 50)
print("2. SIMILARITY POSTPROCESSOR (score filtering)")
print("=" * 50)

postprocessor = SimilarityPostprocessor(similarity_cutoff=0.4)
filtered_nodes = postprocessor.postprocess_nodes(nodes)
print(f"After filtering (cutoff=0.4): {len(filtered_nodes)} nodes remain")

# ── 3. Response Synthesizer Modes ───────────────────────────
print("\n" + "=" * 50)
print("3. RESPONSE SYNTHESIZER MODES")
print("=" * 50)

query = "What AWS services are best for running code without managing servers?"

# Mode 1: COMPACT (default) — fits all context in one LLM call
compact_synth  = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
compact_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=compact_synth)
print(f"\n[COMPACT] {compact_engine.query(query).response[:200]}")

# Mode 2: TREE_SUMMARIZE — builds a tree of summaries (good for long docs)
tree_synth  = get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE)
tree_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=tree_synth)
print(f"\n[TREE_SUMMARIZE] {tree_engine.query(query).response[:200]}")

# Mode 3: NO_TEXT — returns only source nodes, no LLM synthesis
no_text_synth  = get_response_synthesizer(response_mode=ResponseMode.NO_TEXT)
no_text_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=no_text_synth)
result = no_text_engine.query(query)
print(f"\n[NO_TEXT] Source nodes only: {[n.metadata['service'] for n in result.source_nodes]}")

# ── 4. Full Custom Pipeline with Groq ───────────────────────
print("\n" + "=" * 50)
print("4. CUSTOM PIPELINE WITH GROQ")
print("=" * 50)

groq_llm   = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
groq_synth = get_response_synthesizer(llm=groq_llm, response_mode=ResponseMode.COMPACT)
groq_engine = RetrieverQueryEngine(
    retriever=VectorIndexRetriever(index=index, similarity_top_k=2),
    response_synthesizer=groq_synth,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]
)

questions = [
    "Which AWS service should I use for a NoSQL database?",
    "How do I train ML models on AWS?",
]
for q in questions:
    r = groq_engine.query(q)
    print(f"\nQ: {q}")
    print(f"A: {r.response[:200]}")
    print(f"Sources: {[n.metadata['service'] for n in r.source_nodes]}")
