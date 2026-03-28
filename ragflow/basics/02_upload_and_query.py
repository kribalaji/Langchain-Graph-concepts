# RAGFlow Basics 02: Upload Documents & Query
# RAGFlow = deep document parsing + managed RAG pipeline
# Prereq: RAGFlow running + RAGFLOW_API_KEY set in .env

import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:80")
RAGFLOW_API_KEY  = os.environ.get("RAGFLOW_API_KEY", "")

headers = {
    "Authorization": f"Bearer {RAGFLOW_API_KEY}",
    "Content-Type": "application/json"
}

# ── Helper ───────────────────────────────────────────────────
def api(method: str, path: str, **kwargs) -> dict:
    r = getattr(requests, method)(f"{RAGFLOW_BASE_URL}{path}", headers=headers, **kwargs)
    return r.json() if r.status_code == 200 else {"error": r.text[:200]}

# ── 1. Create Dataset ────────────────────────────────────────
print("=" * 50)
print("1. CREATE DATASET")
print("=" * 50)

result = api("post", "/api/v1/datasets", json={
    "name": "company-faq",
    "description": "Company FAQ knowledge base",
    "chunk_method": "qa",       # QA mode: extracts Q&A pairs automatically
})
dataset_id = result.get("data", {}).get("id", "")
print(f"Dataset ID: {dataset_id}")

# ── 2. Upload a Text Document ────────────────────────────────
print("\n" + "=" * 50)
print("2. UPLOAD DOCUMENT")
print("=" * 50)

# Create a sample FAQ text file to upload
faq_content = """Q: What is your refund policy?
A: We offer full refunds within 30 days of purchase with original receipt.

Q: How do I contact support?
A: Our support team is available 24/7 via chat at support.company.com or email support@company.com.

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, PayPal, and bank transfers.

Q: How long does shipping take?
A: Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days.

Q: Do you offer international shipping?
A: Yes, we ship to over 50 countries. International shipping takes 10-14 business days.
"""

# Save temp file
with open("temp_faq.txt", "w") as f:
    f.write(faq_content)

if dataset_id:
    upload_headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    with open("temp_faq.txt", "rb") as f:
        r = requests.post(
            f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents",
            headers=upload_headers,
            files={"file": ("company_faq.txt", f, "text/plain")}
        )
    doc_result = r.json()
    doc_id = doc_result.get("data", [{}])[0].get("id", "")
    print(f"Document uploaded: {doc_id}")

    # ── 3. Parse Document ────────────────────────────────────
    print("\n" + "=" * 50)
    print("3. PARSE DOCUMENT (chunking)")
    print("=" * 50)

    parse_result = api("post", f"/api/v1/datasets/{dataset_id}/chunks",
                       json={"document_ids": [doc_id]})
    print(f"Parse triggered: {parse_result}")

    # Wait for parsing
    print("Waiting for parsing to complete...")
    time.sleep(5)

    # ── 4. List Chunks ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("4. LIST CHUNKS")
    print("=" * 50)

    chunks_result = api("get", f"/api/v1/datasets/{dataset_id}/documents/{doc_id}/chunks")
    chunks = chunks_result.get("data", {}).get("chunks", [])
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1}: {chunk.get('content', '')[:100]}...")

# ── 5. Create Assistant ──────────────────────────────────────
print("\n" + "=" * 50)
print("5. CREATE ASSISTANT (RAG Chatbot)")
print("=" * 50)

if dataset_id:
    assistant_result = api("post", "/api/v1/assistants", json={
        "name": "FAQ Assistant",
        "description": "Answers company FAQ questions",
        "dataset_ids": [dataset_id],
        "llm": {
            "model_name": "mistral:7b",    # Ollama local model
            "temperature": 0.1,
            "top_p": 0.3,
        },
        "prompt": {
            "similarity_threshold": 0.2,
            "keywords_similarity_weight": 0.7,
            "top_n": 3,
            "prompt": "Answer the question based on the FAQ content. Be concise and accurate."
        }
    })
    assistant_id = assistant_result.get("data", {}).get("id", "")
    print(f"Assistant ID: {assistant_id}")

    # ── 6. Create Session & Chat ─────────────────────────────
    print("\n" + "=" * 50)
    print("6. CHAT WITH ASSISTANT")
    print("=" * 50)

    if assistant_id:
        session = api("post", f"/api/v1/assistants/{assistant_id}/sessions",
                      json={"name": "test-session"})
        session_id = session.get("data", {}).get("id", "")

        questions = [
            "What is the refund policy?",
            "How can I contact support?",
            "How long does shipping take?",
        ]

        for q in questions:
            chat_result = api("post",
                f"/api/v1/assistants/{assistant_id}/sessions/{session_id}/chat",
                json={"question": q, "stream": False}
            )
            answer = chat_result.get("data", {}).get("answer", "No answer")
            print(f"\nQ: {q}")
            print(f"A: {answer[:200]}")

# Cleanup temp file
import os as _os
if _os.path.exists("temp_faq.txt"):
    _os.remove("temp_faq.txt")
