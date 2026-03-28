# LangChain Basics 01: LLM Chains, Prompts & Output Parsers
# Models: Ollama (local) + Groq (free cloud)
# Use Case: Customer Support Auto-Reply Generator

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ── Model Setup ──────────────────────────────────────────────
# Ollama: fully local, no API key needed
ollama_llm = ChatOllama(model="mistral:7b", temperature=0)

# Groq: free cloud API — get key at https://console.groq.com
groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── 1. Basic LLM Call ────────────────────────────────────────
print("=" * 50)
print("1. BASIC LLM CALL")
print("=" * 50)

# Ollama
response = ollama_llm.invoke("What is LangChain in one sentence?")
print(f"[Ollama] {response.content}")

# Groq
response = groq_llm.invoke("What is LangChain in one sentence?")
print(f"[Groq]   {response.content}")

# ── 2. Prompt Template + LCEL Chain ─────────────────────────
print("\n" + "=" * 50)
print("2. PROMPT TEMPLATE + LCEL CHAIN")
print("=" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent for {company}. Be concise."),
    ("human", "{question}")
])

# Swap models easily — same chain, different LLM
for name, llm in [("Ollama", ollama_llm), ("Groq", groq_llm)]:
    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke({"company": "TechCorp", "question": "How do I reset my password?"})
    print(f"\n[{name}] {reply}")

# ── 3. Structured JSON Output ────────────────────────────────
print("\n" + "=" * 50)
print("3. STRUCTURED JSON OUTPUT")
print("=" * 50)

class SupportTicket(BaseModel):
    category: str = Field(description="Issue category: billing, technical, general")
    priority: str = Field(description="Priority level: low, medium, high")
    summary: str = Field(description="One-line summary of the issue")

parser = JsonOutputParser(pydantic_object=SupportTicket)

classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the customer issue. {format_instructions}"),
    ("human", "{issue}")
])

# Using Groq for structured output (faster)
classify_chain = classify_prompt | groq_llm | parser
ticket = classify_chain.invoke({
    "issue": "My payment was charged twice and I cannot login!",
    "format_instructions": parser.get_format_instructions()
})
print(f"[Groq] Classified Ticket: {ticket}")

# ── 4. Few-Shot Prompting ────────────────────────────────────
print("\n" + "=" * 50)
print("4. FEW-SHOT PROMPTING")
print("=" * 50)

few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate customer complaints into professional responses."),
    ("human", "Your app is broken!"),
    ("ai", "We apologize for the inconvenience. Our team is actively investigating."),
    ("human", "I was charged twice!"),
    ("ai", "We sincerely apologize for the billing error. A full refund will be processed within 3-5 business days."),
    ("human", "{complaint}")
])

# Using Ollama (local) for few-shot
chain = few_shot_prompt | ollama_llm | StrOutputParser()
print(f"[Ollama] {chain.invoke({'complaint': 'The app keeps crashing on my phone!'})}")

# ── 5. Chaining Multiple Steps ───────────────────────────────
print("\n" + "=" * 50)
print("5. MULTI-STEP CHAIN")
print("=" * 50)

# Step 1: Detect language
detect_prompt = ChatPromptTemplate.from_messages([
    ("system", "Detect the language of the text. Return only the language name."),
    ("human", "{text}")
])

# Step 2: Translate to English
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following {language} text to English. Return only the translation."),
    ("human", "{text}")
])

detect_chain = detect_prompt | groq_llm | StrOutputParser()
translate_chain = translate_prompt | groq_llm | StrOutputParser()

text = "Hola, necesito ayuda con mi cuenta."
language = detect_chain.invoke({"text": text})
translation = translate_chain.invoke({"text": text, "language": language})

print(f"[Groq] Original : {text}")
print(f"[Groq] Language : {language}")
print(f"[Groq] Translated: {translation}")
