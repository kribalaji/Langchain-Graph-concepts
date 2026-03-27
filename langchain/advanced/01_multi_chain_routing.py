# LangChain Advanced: Multi-Chain Routing & Parallel Execution
# Use Case: Intelligent Query Router for Enterprise AI Platform

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

# --- 1. Specialized Chains ---
code_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert software engineer. Provide clean, production-ready code with brief explanation."),
        ("human", "{query}")
    ]) | llm | StrOutputParser()
)

analysis_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Provide structured analysis with key insights and recommendations."),
        ("human", "{query}")
    ]) | llm | StrOutputParser()
)

creative_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a creative writer. Be imaginative, engaging, and original."),
        ("human", "{query}")
    ]) | creative_llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise and accurate."),
        ("human", "{query}")
    ]) | llm | StrOutputParser()
)

# --- 2. Router Chain ---
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify the query into exactly one category: code, analysis, creative, general.
Return ONLY the category word."""),
    ("human", "{query}")
])

router_chain = router_prompt | llm | StrOutputParser()

# --- 3. Dynamic Router ---
def route(inputs: dict) -> str:
    query = inputs["query"]
    category = router_chain.invoke({"query": query}).strip().lower()
    print(f"  [Router] Category: {category}")
    routes = {"code": code_chain, "analysis": analysis_chain, "creative": creative_chain}
    return routes.get(category, general_chain).invoke({"query": query})

full_router = RunnableLambda(route)

# --- 4. Test Routing ---
queries = [
    "Write a Python function to find duplicates in a list",
    "Analyze the pros and cons of microservices vs monolithic architecture",
    "Write a short poem about artificial intelligence",
    "What is the capital of France?",
]

print("=== Intelligent Query Router ===")
for q in queries:
    print(f"\nQuery: {q}")
    result = full_router.invoke({"query": q})
    print(f"Response: {result[:200]}...")

# --- 5. Parallel Chain Execution ---
parallel_analysis = RunnableParallel(
    technical=code_chain,
    business=analysis_chain,
    summary=(ChatPromptTemplate.from_messages([
        ("system", "Summarize in 2 sentences."),
        ("human", "{query}")
    ]) | llm | StrOutputParser())
)

print("\n=== Parallel Analysis ===")
topic = {"query": "Explain the benefits of using vector databases for AI applications"}
parallel_result = parallel_analysis.invoke(topic)
for key, val in parallel_result.items():
    print(f"\n[{key.upper()}]: {val[:150]}...")
