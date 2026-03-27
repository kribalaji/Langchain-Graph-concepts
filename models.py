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
# llama3-8b-8192     — fast, great quality
# mixtral-8x7b-32768 — large context window (32k tokens)
# gemma2-9b-it       — Google's Gemma 2

groq_llm     = ChatGroq(model="llama3-8b-8192",     temperature=0)
groq_large   = ChatGroq(model="mixtral-8x7b-32768", temperature=0)  # 32k context

# ── Model Selection Guide ────────────────────────────────────
# Use ollama_llm  → offline work, sensitive data, no rate limits
# Use ollama_fast → quick tests, simple tasks, low RAM
# Use groq_llm    → fast responses, complex reasoning, free tier
# Use groq_large  → long documents, large context needed
