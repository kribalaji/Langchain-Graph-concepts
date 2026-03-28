# LlamaIndex Basics 02: Chat Engine & Conversation Memory
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Stateful Conversational Q&A over Documents

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm         = Ollama(model="mistral:7b", request_timeout=120.0)

# ── 1. Build Index ───────────────────────────────────────────
documents = [
    Document(text="""Python is a high-level programming language created by Guido van Rossum in 1991.
    It emphasizes code readability with significant indentation. Python supports multiple paradigms
    including procedural, object-oriented, and functional programming. It has a large standard library.""",
    metadata={"topic": "python"}),

    Document(text="""Python is widely used in data science, machine learning, web development, and automation.
    Popular frameworks include Django and FastAPI for web, TensorFlow and PyTorch for ML,
    and Pandas and NumPy for data analysis. Python consistently ranks as the most popular language.""",
    metadata={"topic": "python_uses"}),

    Document(text="""Python 3.12 was released in October 2023 with 5% average performance improvement.
    Python 3.13 introduces experimental free-threaded mode (no GIL) for better concurrency.
    The Python Software Foundation manages the language development and releases.""",
    metadata={"topic": "python_versions"}),
]

index = VectorStoreIndex.from_documents(documents)
print("Index built!\n")

# ── 2. Chat Engine Modes ─────────────────────────────────────
print("=" * 50)
print("CHAT ENGINE MODES")
print("=" * 50)
print("""
  condense_question  → rewrites follow-up questions using history (best for RAG)
  context            → retrieves context for every message
  simple             → no retrieval, pure LLM chat with memory
  react              → uses ReAct reasoning with tools
""")

# ── 3. Condense Question Chat Engine (Ollama) ────────────────
print("=" * 50)
print("CONDENSE QUESTION MODE — OLLAMA")
print("=" * 50)

memory      = ChatMemoryBuffer.from_defaults(token_limit=2048)
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    verbose=True
)

conversations = [
    "What is Python?",
    "When was it created?",           # follow-up — needs history
    "What are its main use cases?",
    "Which version added GIL removal?",
]

for msg in conversations:
    print(f"\nUser: {msg}")
    response = chat_engine.chat(msg)
    print(f"Bot : {str(response)[:250]}")

# ── 4. Reset and use Groq ────────────────────────────────────
print("\n" + "=" * 50)
print("CONTEXT MODE — GROQ")
print("=" * 50)

groq_llm    = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
groq_memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

groq_chat = index.as_chat_engine(
    chat_mode="context",
    llm=groq_llm,
    memory=groq_memory,
    system_prompt="You are a Python expert. Answer concisely using the provided context."
)

groq_questions = [
    "What Python frameworks are popular for web development?",
    "What about for machine learning?",   # tests memory
]

for msg in groq_questions:
    print(f"\nUser: {msg}")
    response = groq_chat.chat(msg)
    print(f"Bot : {str(response)[:250]}")

# ── 5. Chat History ──────────────────────────────────────────
print("\n" + "=" * 50)
print("CHAT HISTORY")
print("=" * 50)
history = memory.get()
print(f"Total messages in memory: {len(history)}")
for msg in history[-4:]:
    print(f"  [{msg.role}]: {str(msg.content)[:80]}...")
