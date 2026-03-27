# LangChain Basics: Document Loaders & Text Splitters
# Use Case: Enterprise Document Ingestion Pipeline

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# --- 1. Manual Document Creation (simulating a loader) ---
raw_docs = [
    Document(
        page_content="""LangChain is a framework for developing applications powered by large language models.
        It provides tools for chaining LLM calls, managing prompts, and integrating with external data sources.
        LangChain supports multiple LLM providers including OpenAI, Anthropic, and open-source models.
        The framework includes modules for retrieval-augmented generation (RAG), agents, and memory management.""",
        metadata={"source": "langchain_docs.txt", "page": 1}
    ),
    Document(
        page_content="""RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models.
        It first retrieves relevant documents from a knowledge base, then uses them as context for generation.
        This approach reduces hallucinations and keeps responses grounded in factual data.
        RAG is widely used in enterprise Q&A systems, chatbots, and knowledge management tools.""",
        metadata={"source": "rag_overview.txt", "page": 1}
    )
]

# --- 2. Recursive Character Text Splitter (most common) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(raw_docs)
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} [{chunk.metadata['source']}]:\n{chunk.page_content}")

# --- 3. Token-based Splitter (for token-aware chunking) ---
token_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
token_chunks = token_splitter.split_documents(raw_docs)
print(f"\nToken-based chunks: {len(token_chunks)}")

# --- 4. Metadata-aware splitting ---
def split_with_metadata(docs, chunk_size=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["chunk_total"] = len(split_docs)
    return split_docs

enriched_chunks = split_with_metadata(raw_docs)
print(f"\nEnriched chunks sample metadata: {enriched_chunks[0].metadata}")
