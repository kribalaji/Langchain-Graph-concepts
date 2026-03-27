# LangChain + LangGraph + RAGAS: Complete Learning Path

A comprehensive, production-focused project covering basics to advanced topics with real-world use cases.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

## Project Structure & Learning Path

```
├── langchain/
│   ├── basics/
│   │   ├── 01_llm_chains_prompts.py        → LLM calls, LCEL chains, output parsers
│   │   ├── 02_document_loaders_splitters.py → Document ingestion, chunking strategies
│   │   └── 03_embeddings_vectorstores.py   → Embeddings, FAISS, semantic search
│   ├── intermediate/
│   │   ├── 01_rag_pipeline.py              → ★ Production RAG with citations
│   │   ├── 02_memory_conversation.py       → Multi-session memory management
│   │   └── 03_agents_tools.py             → Custom tools, financial research agent
│   └── advanced/
│       ├── 01_multi_chain_routing.py       → Dynamic routing, parallel chains
│       ├── 02_streaming_async.py           → Streaming, async/await patterns
│       └── 03_advanced_retrieval.py        → HyDE, MMR, multi-query, compression
│
├── langgraph/
│   ├── basics/
│   │   ├── 01_state_graph_nodes_edges.py   → StateGraph, nodes, edges, blog pipeline
│   │   └── 02_message_state_chatbot.py     → Message state, add_messages reducer
│   ├── intermediate/
│   │   ├── 01_conditional_edges.py         → Branching, code review pipeline
│   │   ├── 02_checkpointing_persistence.py → MemorySaver, thread-based sessions
│   │   └── 03_human_in_the_loop.py        → Interrupt, human approval workflow
│   └── advanced/
│       ├── 01_multi_agent_supervisor.py    → ★ Supervisor pattern, market research team
│       ├── 02_react_agent_cycles.py        → ReAct agent, ToolNode, DevOps assistant
│       └── 03_parallel_subgraphs.py        → Send API, fan-out/fan-in, subgraphs
│
└── ragas/
    ├── basics/
    │   ├── 01_core_metrics.py              → Faithfulness, relevancy, precision, recall
    │   └── 02_synthetic_data_generation.py → Auto-generate test datasets
    ├── intermediate/
    │   ├── 01_rag_evaluation_pipeline.py   → ★ Compare RAG configurations
    │   └── 02_retriever_comparison.py      → Evaluate retrieval strategies
    └── advanced/
        ├── 01_custom_metrics_cicd.py       → Custom metrics, CI/CD quality gate
        └── 02_langgraph_ragas_eval_loop.py → Self-optimizing RAG with eval loop
```

## Key Concepts by Topic

### LangChain
| Level | Concept | File |
|-------|---------|------|
| Basics | LCEL chains, prompts, parsers | `01_llm_chains_prompts.py` |
| Basics | Chunking strategies | `02_document_loaders_splitters.py` |
| Basics | Vector stores, semantic search | `03_embeddings_vectorstores.py` |
| Intermediate | RAG with source citations | `01_rag_pipeline.py` |
| Intermediate | Session-based memory | `02_memory_conversation.py` |
| Intermediate | Tool-calling agents | `03_agents_tools.py` |
| Advanced | Query routing | `01_multi_chain_routing.py` |
| Advanced | Streaming & async | `02_streaming_async.py` |
| Advanced | HyDE, MMR, compression | `03_advanced_retrieval.py` |

### LangGraph
| Level | Concept | File |
|-------|---------|------|
| Basics | StateGraph, nodes, edges | `01_state_graph_nodes_edges.py` |
| Basics | Message state, add_messages | `02_message_state_chatbot.py` |
| Intermediate | Conditional edges, cycles | `01_conditional_edges.py` |
| Intermediate | Checkpointing, persistence | `02_checkpointing_persistence.py` |
| Intermediate | Human-in-the-loop | `03_human_in_the_loop.py` |
| Advanced | Multi-agent supervisor | `01_multi_agent_supervisor.py` |
| Advanced | ReAct + ToolNode | `02_react_agent_cycles.py` |
| Advanced | Parallel subgraphs, Send API | `03_parallel_subgraphs.py` |

### RAGAS
| Level | Concept | File |
|-------|---------|------|
| Basics | 4 core metrics explained | `01_core_metrics.py` |
| Basics | Synthetic test generation | `02_synthetic_data_generation.py` |
| Intermediate | RAG config comparison | `01_rag_evaluation_pipeline.py` |
| Intermediate | Retriever benchmarking | `02_retriever_comparison.py` |
| Advanced | Custom metrics + CI/CD gate | `01_custom_metrics_cicd.py` |
| Advanced | Self-optimizing eval loop | `02_langgraph_ragas_eval_loop.py` |

## RAGAS Metrics Quick Reference

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Answer grounded in context? (anti-hallucination) | > 0.85 |
| **Answer Relevancy** | Answer relevant to question? | > 0.80 |
| **Context Precision** | Retrieved chunks are useful? | > 0.75 |
| **Context Recall** | Context contains all needed info? | > 0.75 |

## Most Critical Market Use Cases

1. **Enterprise RAG Q&A** → `langchain/intermediate/01_rag_pipeline.py`
2. **Multi-Agent Workflows** → `langgraph/advanced/01_multi_agent_supervisor.py`
3. **RAG Evaluation & CI/CD** → `ragas/advanced/01_custom_metrics_cicd.py`
4. **Human-in-the-Loop Approval** → `langgraph/intermediate/03_human_in_the_loop.py`
5. **Self-Optimizing RAG** → `ragas/advanced/02_langgraph_ragas_eval_loop.py`

## Environment Variables

```env
OPENAI_API_KEY=your_key_here
```
