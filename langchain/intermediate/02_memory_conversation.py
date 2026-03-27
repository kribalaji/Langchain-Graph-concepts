# LangChain Intermediate: Memory & Conversation Management
# Use Case: Persistent AI Sales Assistant with Context

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- 1. In-Memory Session Store ---
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# --- 2. Sales Assistant Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Alex, an AI sales assistant for TechStore.
- Remember customer preferences and previous interactions
- Proactively suggest relevant products based on conversation history
- Be concise and helpful"""),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# --- 3. Chain with Message History ---
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- 4. Simulate Multi-turn Sales Conversation ---
session_id = "customer_001"
config = {"configurable": {"session_id": session_id}}

conversations = [
    "Hi, I'm looking for a laptop for video editing",
    "My budget is around $1500-2000",
    "I prefer Mac over Windows",
    "What was my budget again?",  # Tests memory recall
]

print("=== Sales Assistant Conversation ===")
for user_msg in conversations:
    response = chain_with_history.invoke({"input": user_msg}, config=config)
    print(f"\nCustomer: {user_msg}")
    print(f"Alex: {response}")

# --- 5. Inspect Stored History ---
history = get_session_history(session_id)
print(f"\n=== Session History ({len(history.messages)} messages) ===")
for msg in history.messages:
    role = "Customer" if isinstance(msg, HumanMessage) else "Alex"
    print(f"{role}: {msg.content[:80]}...")

# --- 6. Multi-session Isolation ---
session_store["customer_002"] = InMemoryChatMessageHistory()
session_store["customer_002"].add_message(HumanMessage(content="I want headphones under $200"))

response_002 = chain_with_history.invoke(
    {"input": "What was I looking for?"},
    config={"configurable": {"session_id": "customer_002"}}
)
print(f"\n=== Customer 002 (different session) ===\n{response_002}")
