# LangChain Basics 02: Document Loaders & Text Splitters
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Enterprise Document Ingestion Pipeline

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

load_dotenv()

ollama_llm = ChatOllama(model="mistral:7b", temperature=0)
groq_llm   = ChatGroq(model="llama3-8b-8192", temperature=0)

# ── Sample Documents ─────────────────────────────────────────
raw_docs = [
    Document(
        page_content="""LangChain is a framework for developing applications powered by large language models.
It provides tools for chaining LLM calls, managing prompts, and integrating with external data.
LangChain supports multiple LLM providers including Ollama, Groq, and HuggingFace.
The framework includes modules for RAG, agents, and memory management.
LCEL (LangChain Expression Language) allows composing chains declaratively using the pipe operator.""",
        metadata={"source": "langchain_docs.txt", "page": 1}
    ),
    Document(
        page_content="""RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models.
It first retrieves relevant documents from a knowledge base, then uses them as context for generation.
This approach reduces hallucinations and keeps responses grounded in factual data.
RAG is widely used in enterprise Q&A systems, chatbots, and knowledge management tools.
Key components: document loader, text splitter, embeddings, vector store, retriever, and LLM.""",
        metadata={"source": "rag_overview.txt", "page": 1}
    ),
]

# ── 1. Recursive Character Text Splitter ────────────────────
print("=" * 50)
print("1. RECURSIVE CHARACTER SPLITTER")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(raw_docs)
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} [{chunk.metadata['source']}]:\n{chunk.page_content}")

# ── 2. Token-based Splitter ──────────────────────────────────
print("\n" + "=" * 50)
print("2. TOKEN-BASED SPLITTER")
print("=" * 50)

token_splitter = TokenTextSplitter(chunk_size=60, chunk_overlap=10)
token_chunks = token_splitter.split_documents(raw_docs)
print(f"Token-based chunks: {len(token_chunks)}")
for i, chunk in enumerate(token_chunks[:3]):
    print(f"  Chunk {i+1}: {chunk.page_content[:80]}...")

# ── 3. Markdown Header Splitter ──────────────────────────────
print("\n" + "=" * 50)
print("3. MARKDOWN HEADER SPLITTER")
print("=" * 50)

markdown_text = """# LangChain Guide

## Installation
Install LangChain using pip. You also need langchain-ollama for local models.

## Core Concepts
LangChain uses LCEL to compose chains. Each chain is a sequence of runnables.

## Getting Started
Create a simple chain by combining a prompt, LLM, and output parser.
"""

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")]
)
md_chunks = md_splitter.split_text(markdown_text)
for chunk in md_chunks:
    print(f"\nMetadata: {chunk.metadata}")
    print(f"Content : {chunk.page_content[:80]}")

# ── 4. Metadata-Enriched Splitting ──────────────────────────
print("\n" + "=" * 50)
print("4. METADATA-ENRICHED CHUNKS")
print("=" * 50)

def split_with_metadata(docs, chunk_size=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["chunk_total"] = len(split_docs)
    return split_docs

enriched = split_with_metadata(raw_docs)
print(f"Enriched chunks: {len(enriched)}")
print(f"Sample metadata: {enriched[0].metadata}")

# ── 5. Summarize Chunks with LLM ────────────────────────────
print("\n" + "=" * 50)
print("5. SUMMARIZE CHUNK WITH LLM")
print("=" * 50)

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one sentence."),
    ("human", "{text}")
])

# Ollama for local summarization
ollama_chain = summarize_prompt | ollama_llm | StrOutputParser()
groq_chain   = summarize_prompt | groq_llm   | StrOutputParser()

sample_chunk = chunks[0].page_content
print(f"Chunk: {sample_chunk}")
print(f"\n[Ollama] Summary: {ollama_chain.invoke({'text': sample_chunk})}")
print(f"[Groq]   Summary: {groq_chain.invoke({'text': sample_chunk})}")
