# LangChain Advanced: Advanced Retrieval Strategies
# Use Case: High-Accuracy Legal Document Search System

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Legal Document Knowledge Base ---
legal_docs = [
    Document(page_content="Section 12.3: Termination for Cause. Either party may terminate this agreement immediately upon written notice if the other party materially breaches any provision and fails to cure such breach within 30 days.", metadata={"doc": "contract_A", "section": "12.3"}),
    Document(page_content="Section 8.1: Intellectual Property. All work product created by Contractor under this agreement shall be considered work-for-hire and ownership shall vest in Client upon full payment.", metadata={"doc": "contract_A", "section": "8.1"}),
    Document(page_content="Section 15.2: Limitation of Liability. In no event shall either party be liable for indirect, incidental, or consequential damages exceeding the total fees paid in the preceding 12 months.", metadata={"doc": "contract_B", "section": "15.2"}),
    Document(page_content="Section 6.4: Confidentiality. Recipient shall protect Discloser's confidential information using the same degree of care as its own confidential information, but no less than reasonable care, for 5 years.", metadata={"doc": "contract_B", "section": "6.4"}),
    Document(page_content="Section 9.2: Dispute Resolution. All disputes shall first be subject to good-faith negotiation for 30 days, then mediation, and finally binding arbitration under AAA Commercial Rules.", metadata={"doc": "contract_C", "section": "9.2"}),
]

vectorstore = FAISS.from_documents(legal_docs, embeddings)

# --- 1. MMR Retrieval (Maximal Marginal Relevance - reduces redundancy) ---
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)
print("=== MMR Retrieval ===")
mmr_results = mmr_retriever.invoke("What happens when a contract is violated?")
for doc in mmr_results:
    print(f"  [{doc.metadata['section']}]: {doc.page_content[:80]}...")

# --- 2. Multi-Query Retrieval (generates multiple query variations) ---
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)
print("\n=== Multi-Query Retrieval ===")
mq_results = multi_query_retriever.invoke("Who owns the code I write for a client?")
for doc in mq_results:
    print(f"  [{doc.metadata['section']}]: {doc.page_content[:80]}...")

# --- 3. Contextual Compression (extracts only relevant parts) ---
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
print("\n=== Contextual Compression ===")
compressed = compression_retriever.invoke("What is the liability cap?")
for doc in compressed:
    print(f"  Compressed: {doc.page_content}")

# --- 4. HyDE - Hypothetical Document Embeddings ---
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a hypothetical legal contract clause that would answer this question. Be specific and formal."),
    ("human", "{question}")
])

def hyde_retrieve(question: str, k: int = 2):
    hypothetical_doc = (hyde_prompt | llm | StrOutputParser()).invoke({"question": question})
    return vectorstore.similarity_search(hypothetical_doc, k=k)

print("\n=== HyDE Retrieval ===")
hyde_results = hyde_retrieve("How long must confidential information be protected?")
for doc in hyde_results:
    print(f"  [{doc.metadata['section']}]: {doc.page_content[:80]}...")

# --- 5. Full Advanced RAG Pipeline ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal assistant. Answer based strictly on the provided contract clauses.\n\nClauses:\n{context}"),
    ("human", "{question}")
])

advanced_rag = (
    {"context": mmr_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": lambda x: x}
    | rag_prompt | llm | StrOutputParser()
)

print("\n=== Advanced Legal RAG ===")
print(advanced_rag.invoke("What are my options if the other party breaches the contract?"))
