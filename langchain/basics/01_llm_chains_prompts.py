# LangChain Basics: LLM Chains, Prompts, Output Parsers
# Use Case: Customer Support Auto-Reply Generator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. Basic LLM Call ---
response = llm.invoke("What is LangChain in one sentence?")
print("Basic LLM:", response.content)

# --- 2. Prompt Template + Chain (LCEL) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent for {company}."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()
reply = chain.invoke({"company": "TechCorp", "question": "How do I reset my password?"})
print("\nCustomer Support Reply:\n", reply)

# --- 3. Structured Output with JSON Parser ---
class SupportTicket(BaseModel):
    category: str = Field(description="Issue category: billing, technical, general")
    priority: str = Field(description="Priority: low, medium, high")
    summary: str = Field(description="One-line summary of the issue")

parser = JsonOutputParser(pydantic_object=SupportTicket)

classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the customer issue. {format_instructions}"),
    ("human", "{issue}")
])

classify_chain = classify_prompt | llm | parser
ticket = classify_chain.invoke({
    "issue": "My payment was charged twice last night and I can't login!",
    "format_instructions": parser.get_format_instructions()
})
print("\nClassified Ticket:", ticket)

# --- 4. Few-Shot Prompting ---
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate customer complaints to professional responses."),
    ("human", "Your app is broken!"),
    ("ai", "We apologize for the inconvenience. Our team is actively investigating the issue."),
    ("human", "I was charged twice!"),
    ("ai", "We sincerely apologize for the billing error. We will process a full refund within 3-5 business days."),
    ("human", "{complaint}")
])

few_shot_chain = few_shot_prompt | llm | StrOutputParser()
print("\nFew-Shot Response:", few_shot_chain.invoke({"complaint": "The app keeps crashing on my phone!"}))
