# models.py — Shared model configuration for all examples
# Import this in any file: from models import ollama_llm, groq_llm, embeddings

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

# ── Ollama (local, no API key) ───────────────────────────────
# mistral:7b  — best quality, 4.4GB
# qwen2.5:3b  — fastest, 1.9GB, good for simple tasks
# qwen3:4b    — balanced, 2.5GB
# llama2      — general purpose, 3.8GB

ollama_llm   = ChatOllama(model="mistral:7b",  temperature=0)
ollama_fast  = ChatOllama(model="qwen2.5:3b",  temperature=0)   # lightweight
embeddings   = OllamaEmbeddings(model="nomic-embed-text")        # local embeddings

# ── Groq (free cloud, needs GROQ_API_KEY) ───────────────────
# llama-3.1-8b-instant    — fast, great quality (replaces llama3-8b)
# llama-3.3-70b-versatile — powerful, best quality
# qwen/qwen3-32b           — Alibaba Qwen3

groq_llm     = ChatGroq(model="llama-3.1-8b-instant",   temperature=0)  # fast, free
groq_large   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  # powerful

# ── Model Selection Guide ────────────────────────────────────
# Use ollama_llm  → offline work, sensitive data, no rate limits
# Use ollama_fast → quick tests, simple tasks, low RAM
# Use groq_llm    → fast responses, complex reasoning, free tier
# Use groq_large  → long documents, large context needed
