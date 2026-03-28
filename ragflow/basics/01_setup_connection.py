# RAGFlow Basics 01: Setup, Connection & Health Check
# RAGFlow = open-source RAG engine with deep document parsing
# Run first: docker-compose -f ragflow/docker-compose.yml up -d

"""
RAGFlow Key Concepts:
─────────────────────────────────────────────────────────────
  Dataset    → collection of uploaded documents (like a knowledge base)
  Document   → uploaded file (PDF, DOCX, TXT, Excel, HTML, etc.)
  Chunk      → parsed piece of a document (RAGFlow does deep parsing)
  Assistant  → RAG chatbot configured with datasets + LLM settings
  Session    → conversation thread with an assistant
  API Key    → generated from RAGFlow UI → Settings → API Keys

RAGFlow vs LangChain RAG:
─────────────────────────────────────────────────────────────
  LangChain  → you build the pipeline (loader → splitter → embed → store)
  RAGFlow    → fully managed pipeline via UI + API, deep PDF/table parsing
─────────────────────────────────────────────────────────────
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:80")
RAGFLOW_API_KEY  = os.environ.get("RAGFLOW_API_KEY", "")

headers = {
    "Authorization": f"Bearer {RAGFLOW_API_KEY}",
    "Content-Type": "application/json"
}

# ── 1. Health Check ──────────────────────────────────────────
print("=" * 50)
print("1. RAGFLOW HEALTH CHECK")
print("=" * 50)

def check_health():
    try:
        r = requests.get(f"{RAGFLOW_BASE_URL}/api/v1/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ RAGFlow is running at {RAGFLOW_BASE_URL}")
            return True
        else:
            print(f"⚠️  RAGFlow responded with status: {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to RAGFlow at {RAGFLOW_BASE_URL}")
        print("   Run: docker-compose -f ragflow/docker-compose.yml up -d")
        return False

is_running = check_health()

# ── 2. List Datasets ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. LIST DATASETS (Knowledge Bases)")
print("=" * 50)

if is_running and RAGFLOW_API_KEY:
    r = requests.get(f"{RAGFLOW_BASE_URL}/api/v1/datasets", headers=headers)
    if r.status_code == 200:
        datasets = r.json().get("data", {}).get("datasets", [])
        print(f"Found {len(datasets)} datasets:")
        for ds in datasets:
            print(f"  - [{ds['id']}] {ds['name']} | docs: {ds.get('document_count', 0)}")
    else:
        print(f"Error: {r.status_code} — {r.text[:100]}")
else:
    print("Skipping — RAGFlow not running or API key not set")
    print("\nTo get your API key:")
    print("  1. Open http://localhost:80")
    print("  2. Register/Login")
    print("  3. Go to Settings → API Keys → Create")
    print("  4. Add to .env: RAGFLOW_API_KEY=ragflow-xxxx")

# ── 3. Create a Dataset ──────────────────────────────────────
print("\n" + "=" * 50)
print("3. CREATE DATASET")
print("=" * 50)

def create_dataset(name: str, description: str = "") -> dict:
    payload = {
        "name": name,
        "description": description,
        "embedding_model": "nomic-embed-text",   # use local Ollama embedding
        "chunk_method": "naive",                  # options: naive, manual, qa, table
    }
    r = requests.post(f"{RAGFLOW_BASE_URL}/api/v1/datasets", headers=headers, json=payload)
    if r.status_code == 200:
        data = r.json().get("data", {})
        print(f"✅ Dataset created: {data.get('name')} (id: {data.get('id')})")
        return data
    else:
        print(f"❌ Failed: {r.status_code} — {r.text[:150]}")
        return {}

if is_running and RAGFLOW_API_KEY:
    dataset = create_dataset("tech-knowledge-base", "Technology Q&A knowledge base")
else:
    print("Skipping — RAGFlow not running")
    print("Example payload:")
    print("""  POST /api/v1/datasets
  {
    "name": "tech-knowledge-base",
    "embedding_model": "nomic-embed-text",
    "chunk_method": "naive"
  }""")

# ── 4. RAGFlow Chunk Methods Explained ──────────────────────
print("\n" + "=" * 50)
print("4. CHUNK METHODS REFERENCE")
print("=" * 50)
print("""
  naive      → simple fixed-size chunking (like LangChain RecursiveCharacterTextSplitter)
  manual     → user-defined chunk boundaries
  qa         → extracts Q&A pairs from documents
  table      → specialized for Excel/CSV table parsing
  paper      → optimized for academic papers (title, abstract, sections)
  book       → chapter-aware chunking for long books
  laws       → structured parsing for legal documents
  presentation → slide-aware chunking for PowerPoint files
""")
