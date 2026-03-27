# LangChain Basics 03: Embeddings & Vector Stores
# Embeddings: Ollama nomic-embed-text (local, free)
# Use Case: Semantic Product Search

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ── Embeddings Setup ─────────────────────────────────────────
# nomic-embed-text is already pulled (274MB, fast, high quality)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ── 1. Embed a Single Text ───────────────────────────────────
print("=" * 50)
print("1. SINGLE TEXT EMBEDDING")
print("=" * 50)

vector = embeddings.embed_query("What is LangChain?")
print(f"Embedding dimensions: {len(vector)}")
print(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")

# ── 2. Product Catalog Documents ────────────────────────────
print("\n" + "=" * 50)
print("2. BUILD FAISS VECTOR STORE")
print("=" * 50)

products = [
    Document(page_content="iPhone 15 Pro - 6.1 inch display, A17 Pro chip, 48MP camera, titanium design", metadata={"id": "p1", "category": "smartphone", "price": 999}),
    Document(page_content="Samsung Galaxy S24 - 6.2 inch AMOLED, Snapdragon 8 Gen 3, 50MP camera, AI features", metadata={"id": "p2", "category": "smartphone", "price": 799}),
    Document(page_content="MacBook Pro 14 - M3 Pro chip, 14 inch Liquid Retina, 18hr battery, 16GB RAM", metadata={"id": "p3", "category": "laptop", "price": 1999}),
    Document(page_content="Dell XPS 15 - Intel Core i9, 15.6 inch OLED, NVIDIA RTX 4060, 32GB RAM", metadata={"id": "p4", "category": "laptop", "price": 2199}),
    Document(page_content="Sony WH-1000XM5 - Industry-leading noise cancellation, 30hr battery, multipoint", metadata={"id": "p5", "category": "headphones", "price": 349}),
    Document(page_content="iPad Pro 12.9 - M2 chip, Liquid Retina XDR, Apple Pencil support, 5G", metadata={"id": "p6", "category": "tablet", "price": 1099}),
]

vectorstore = FAISS.from_documents(products, embeddings)
print(f"Vector store created: {vectorstore.index.ntotal} vectors")

# ── 3. Similarity Search ─────────────────────────────────────
print("\n" + "=" * 50)
print("3. SIMILARITY SEARCH")
print("=" * 50)

queries = [
    "best phone with great camera",
    "powerful laptop for developers",
    "wireless headphones with noise cancellation",
]

for query in queries:
    results = vectorstore.similarity_search(query, k=2)
    print(f"\nQuery: '{query}'")
    for doc in results:
        print(f"  → {doc.page_content[:60]}... | ${doc.metadata['price']}")

# ── 4. Similarity Search with Score ─────────────────────────
print("\n" + "=" * 50)
print("4. SEARCH WITH SIMILARITY SCORES")
print("=" * 50)

results_with_score = vectorstore.similarity_search_with_score("tablet for creative work", k=3)
for doc, score in results_with_score:
    print(f"  Score: {score:.4f} | {doc.page_content[:55]}...")

# ── 5. Metadata Filtering ────────────────────────────────────
print("\n" + "=" * 50)
print("5. METADATA FILTERING")
print("=" * 50)

laptop_results = vectorstore.similarity_search(
    "machine for video editing",
    k=2,
    filter={"category": "laptop"}
)
print("Filtered to laptops only:")
for doc in laptop_results:
    print(f"  → {doc.page_content[:60]}...")

# ── 6. Save & Reload Vector Store ───────────────────────────
print("\n" + "=" * 50)
print("6. SAVE & RELOAD VECTOR STORE")
print("=" * 50)

vectorstore.save_local("product_catalog_index")
loaded = FAISS.load_local(
    "product_catalog_index", embeddings,
    allow_dangerous_deserialization=True
)
print(f"Reloaded store vectors: {loaded.index.ntotal}")

# Quick test on reloaded store
result = loaded.similarity_search("noise cancelling headphones", k=1)
print(f"Test query result: {result[0].page_content[:60]}...")
