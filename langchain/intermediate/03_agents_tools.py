# LangChain Intermediate: Agents & Tools
# Use Case: AI Financial Research Agent with Real-time Tools

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. Define Custom Tools ---
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    # Simulated data (replace with real API like yfinance)
    prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 415.30, "AMZN": 185.60, "NVDA": 875.40}
    price = prices.get(ticker.upper())
    return f"{ticker.upper()}: ${price}" if price else f"Ticker {ticker} not found"

@tool
def get_company_news(company: str) -> str:
    """Get recent news headlines for a company."""
    news = {
        "apple": "Apple launches AI features in iOS 18; Q3 revenue beats expectations by 3%",
        "google": "Google DeepMind releases Gemini Ultra 2.0; antitrust case ongoing",
        "microsoft": "Microsoft Azure AI revenue up 29% YoY; Copilot adoption surges",
        "nvidia": "NVIDIA H200 GPU demand exceeds supply; data center revenue hits record $22B",
    }
    return news.get(company.lower(), f"No recent news found for {company}")

@tool
def calculate_portfolio_value(holdings: str) -> str:
    """Calculate total portfolio value. Input format: 'TICKER:SHARES,TICKER:SHARES'
    Example: 'AAPL:10,MSFT:5'"""
    prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 415.30, "AMZN": 185.60, "NVDA": 875.40}
    total = 0
    breakdown = []
    for item in holdings.split(","):
        ticker, shares = item.strip().split(":")
        ticker = ticker.upper()
        price = prices.get(ticker, 0)
        value = price * int(shares)
        total += value
        breakdown.append(f"{ticker}: {shares} shares × ${price} = ${value:,.2f}")
    return "\n".join(breakdown) + f"\nTotal Portfolio Value: ${total:,.2f}"

@tool
def compare_stocks(tickers: str) -> str:
    """Compare multiple stocks. Input: comma-separated tickers like 'AAPL,MSFT,GOOGL'"""
    data = {
        "AAPL": {"price": 189.50, "pe": 28.5, "ytd": "+12%"},
        "MSFT": {"price": 415.30, "pe": 35.2, "ytd": "+18%"},
        "GOOGL": {"price": 175.20, "pe": 24.1, "ytd": "+8%"},
        "NVDA": {"price": 875.40, "pe": 65.3, "ytd": "+145%"},
    }
    result = []
    for t in tickers.split(","):
        t = t.strip().upper()
        if t in data:
            d = data[t]
            result.append(f"{t}: Price=${d['price']}, P/E={d['pe']}, YTD={d['ytd']}")
    return "\n".join(result)

# --- 2. Agent Setup ---
tools = [get_stock_price, get_company_news, calculate_portfolio_value, compare_stocks]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial research assistant. Use tools to provide accurate, data-driven insights."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# --- 3. Run Agent Queries ---
queries = [
    "What's the current price of NVIDIA and any recent news about them?",
    "I have 10 AAPL, 5 MSFT, and 3 NVDA shares. What's my portfolio worth?",
    "Compare AAPL, MSFT, and GOOGL stocks and tell me which looks best",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    result = executor.invoke({"input": query})
    print(f"Answer: {result['output']}")
