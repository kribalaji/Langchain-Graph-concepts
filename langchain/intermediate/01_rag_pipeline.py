# LangChain Intermediate: Production RAG Pipeline
# Use Case: Enterprise Knowledge Base Q&A (most demanded market use case)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Knowledge Base Setup ---
docs = [
    Document(page_content="Our refund policy allows returns within 30 days of purchase with original receipt. Digital products are non-refundable.", metadata={"source": "policy.txt"}),
    Document(page_content="Premium members get free shipping on all orders over $25. Standard members pay $4.99 for shipping.", metadata={"source": "shipping.txt"}),
    Document(page_content="To reset your password: go to Settings > Security > Reset Password. You'll receive an email within 5 minutes.", metadata={"source": "support.txt"}),
    Document(page_content="Our customer support is available 24/7 via chat. Phone support is available Mon-Fri 9AM-6PM EST.", metadata={"source": "contact.txt"}),
    Document(page_content="Premium subscription costs $9.99/month or $99/year. It includes unlimited storage, priority support, and advanced analytics.", metadata={"source": "pricing.txt"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 2. RAG Prompt ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant.
Answer ONLY based on the provided context. If the answer is not in the context, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(f"[{doc.metadata['source']}]: {doc.page_content}" for doc in docs)

# --- 3. Basic RAG Chain ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("=== Basic RAG ===")
print(rag_chain.invoke("What is the refund policy?"))

# --- 4. RAG with Source Citations ---
rag_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=(retriever | (lambda docs: [d.metadata["source"] for d in docs]))
)

result = rag_with_sources.invoke("How much does premium cost?")
print("\n=== RAG with Sources ===")
print("Answer:", result["answer"])
print("Sources:", result["sources"])

# --- 5. Conversational RAG (with history) ---
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only.\n\nContext: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

conv_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda _: []}
    | conv_prompt | llm | StrOutputParser()
)

chat_history = []
questions = ["What are the support hours?", "Can I contact them on weekends?"]
for q in questions:
    answer = conv_chain.invoke(q)
    chat_history.extend([HumanMessage(content=q), AIMessage(content=answer)])
    print(f"\nQ: {q}\nA: {answer}")
