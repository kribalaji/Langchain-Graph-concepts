# LangChain Advanced: Streaming & Async Pipelines
# Use Case: Real-time AI Content Generation Platform

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

# --- 1. Synchronous Streaming ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical blog writer. Write engaging, informative content."),
    ("human", "Write a short intro paragraph about: {topic}")
])

chain = prompt | llm | StrOutputParser()

print("=== Sync Streaming ===")
print("Topic: Building RAG Applications\n")
for chunk in chain.stream({"topic": "Building RAG Applications with LangChain"}):
    print(chunk, end="", flush=True)
print("\n")

# --- 2. Async Streaming ---
async def stream_async(topic: str):
    print(f"=== Async Streaming: {topic} ===")
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
    print("\n")

# --- 3. Parallel Async Generation ---
async def generate_parallel(topics: list[str]):
    print("=== Parallel Async Generation ===")
    tasks = [chain.ainvoke({"topic": t}) for t in topics]
    results = await asyncio.gather(*tasks)
    for topic, result in zip(topics, results):
        print(f"\n[{topic}]:\n{result[:200]}...")

# --- 4. Streaming with Callbacks ---
from langchain_core.callbacks import StreamingStdOutCallbackHandler

streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

print("=== Streaming with Callback Handler ===")
streaming_chain = prompt | streaming_llm | StrOutputParser()
streaming_chain.invoke({"topic": "LangGraph for multi-agent systems"})

# --- 5. Run Async Examples ---
async def main():
    await stream_async("Vector databases in production")
    await generate_parallel([
        "LangChain memory management",
        "RAGAS evaluation metrics",
        "LangGraph state machines"
    ])

asyncio.run(main())
