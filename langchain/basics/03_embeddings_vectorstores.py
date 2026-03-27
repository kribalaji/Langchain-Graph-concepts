# LangChain Basics: Embeddings & Vector Stores
# Use Case: Semantic Search for Product Catalog

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Product Catalog Documents ---
products = [
    Document(page_content="iPhone 15 Pro - 6.1 inch display, A17 Pro chip, 48MP camera, titanium design", metadata={"id": "p1", "category": "smartphone", "price": 999}),
    Document(page_content="Samsung Galaxy S24 - 6.2 inch AMOLED, Snapdragon 8 Gen 3, 50MP camera, AI features", metadata={"id": "p2", "category": "smartphone", "price": 799}),
    Document(page_content="MacBook Pro 14 - M3 Pro chip, 14 inch Liquid Retina, 18hr battery, 16GB RAM", metadata={"id": "p3", "category": "laptop", "price": 1999}),
    Document(page_content="Dell XPS 15 - Intel Core i9, 15.6 inch OLED, NVIDIA RTX 4060, 32GB RAM", metadata={"id": "p4", "category": "laptop", "price": 2199}),
    Document(page_content="Sony WH-1000XM5 - Industry-leading noise cancellation, 30hr battery, multipoint connection", metadata={"id": "p5", "category": "headphones", "price": 349}),
]

# --- 2. Build FAISS Vector Store ---
vectorstore = FAISS.from_documents(products, embeddings)
print("Vector store created with", vectorstore.index.ntotal, "vectors")

# --- 3. Similarity Search ---
query = "best phone with great camera"
results = vectorstore.similarity_search(query, k=2)
print(f"\nTop 2 results for '{query}':")
for doc in results:
    print(f"  - {doc.page_content[:60]}... | Price: ${doc.metadata['price']}")

# --- 4. Similarity Search with Score ---
results_with_score = vectorstore.similarity_search_with_score(query, k=3)
print(f"\nResults with similarity scores:")
for doc, score in results_with_score:
    print(f"  Score: {score:.4f} | {doc.page_content[:50]}...")

# --- 5. Metadata Filtering ---
laptop_results = vectorstore.similarity_search(
    "powerful machine for developers",
    k=2,
    filter={"category": "laptop"}
)
print(f"\nFiltered laptop results:")
for doc in laptop_results:
    print(f"  - {doc.page_content[:60]}...")

# --- 6. Save and Load Vector Store ---
vectorstore.save_local("product_catalog_index")
loaded_store = FAISS.load_local("product_catalog_index", embeddings, allow_dangerous_deserialization=True)
print(f"\nLoaded store vectors: {loaded_store.index.ntotal}")
