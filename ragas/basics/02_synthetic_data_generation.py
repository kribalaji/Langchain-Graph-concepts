# RAGAS Basics: Synthetic Test Data Generation
# Use Case: Auto-generate evaluation datasets from your documents

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

load_dotenv()

# --- 1. Source Documents ---
docs = [
    Document(
        page_content="""LangChain is an open-source framework for building LLM-powered applications.
        It provides abstractions for chains, agents, memory, and retrieval. LangChain supports
        integration with over 50 LLM providers and 100+ tools. The framework uses LCEL
        (LangChain Expression Language) for composing chains declaratively.""",
        metadata={"source": "langchain_overview.txt"}
    ),
    Document(
        page_content="""LangGraph extends LangChain with graph-based orchestration for multi-agent systems.
        It supports stateful workflows, cycles, and human-in-the-loop interactions. LangGraph uses
        a StateGraph where nodes are functions and edges define flow. It includes built-in
        checkpointing for persistence and supports both synchronous and async execution.""",
        metadata={"source": "langgraph_overview.txt"}
    ),
    Document(
        page_content="""RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework
        for RAG pipelines. It measures faithfulness, answer relevancy, context precision, and
        context recall. RAGAS can generate synthetic test datasets from documents and supports
        custom metrics. It integrates with LangChain and LlamaIndex pipelines.""",
        metadata={"source": "ragas_overview.txt"}
    ),
]

# --- 2. Initialize Generator ---
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings
)

# --- 3. Generate Synthetic Test Set ---
print("=== Generating Synthetic Test Dataset ===")
testset = generator.generate_with_langchain_docs(
    documents=docs,
    test_size=9,
    distributions={
        simple: 0.5,        # Simple factual questions
        reasoning: 0.3,     # Multi-step reasoning questions
        multi_context: 0.2  # Questions requiring multiple docs
    }
)

# --- 4. Inspect Generated Data ---
df = testset.to_pandas()
print(f"\nGenerated {len(df)} test cases")
print("\nColumns:", df.columns.tolist())
print("\n=== Sample Questions ===")
for i, row in df.head(5).iterrows():
    print(f"\n[{row.get('evolution_type', 'N/A')}] Q: {row['question']}")
    print(f"  Ground Truth: {str(row['ground_truth'])[:100]}...")

# --- 5. Save for Reuse ---
df.to_csv("synthetic_testset.csv", index=False)
print(f"\nSaved to synthetic_testset.csv")
