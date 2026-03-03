# rag-pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline built for clarity, extensibility, and real-world applicability. This project demonstrates core RAG concepts including document ingestion, chunking, embedding, vector retrieval, and LLM-based generation.

---

## Overview

RAG is a technique that enhances LLM responses by retrieving relevant context from a knowledge base before generating an answer. Rather than relying solely on the model's training data, the pipeline fetches the most relevant document chunks at query time and passes them as context to the LLM.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Multiple (OpenAI GPT-4o, Anthropic Claude, etc.) |
| Embeddings | OpenAI / HuggingFace |
| Vector Store | Chroma (local) |
| Framework | Python |
| Demo UI | Gradio |

---

## Repo Structure

```
rag-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── embedding/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── store.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── generator.py
│   └── pipeline.py
├── evaluation/
│   └── eval.py
├── notebooks/
│   └── exploration.ipynb
├── ui/
│   └── app.py
├── tests/
│   └── test_pipeline.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

**Prerequisites**
- Python 3.10+
- OpenAI or Anthropic API key

**Installation**

```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

**Run the pipeline**

```bash
python src/pipeline.py
```

**Launch the demo UI**

```bash
gradio ui/app.py
```

---

## Key Design Decisions

**Chunking strategy** — documents are split using a recursive character text splitter with intentional overlap to preserve context across chunk boundaries. Chunk size and overlap are configurable via environment variables.

**Embedding model** — embeddings are abstracted behind a common interface, making it straightforward to swap between OpenAI, HuggingFace, or other providers without changing downstream code.

**Vector store** — Chroma is used for local development due to its zero-infrastructure setup. The vector store interface is designed to be provider-agnostic.

**Multiple LLMs** — the generation layer supports swapping between LLM providers to compare output quality across models on the same retrieval results.

---

## Evaluation

The `evaluation/` module measures retrieval precision and recall, answer faithfulness, answer relevance, and end to end latency per query.

---

## Production Considerations

This project is intentionally scoped for demonstration. In a production system:

- **Vector store** — Chroma would be replaced by a managed service such as Pinecone or Qdrant for scalability and persistence across deployments.
- **Memory** — short term conversational memory would be backed by Redis for persistent, low-latency session storage across multiple users and requests.
- **API layer** — the pipeline would be exposed via a FastAPI service with proper authentication, rate limiting, and async request handling.
- **Frontend** — the Gradio demo would be replaced by a React or Next.js frontend consuming the API.
- **Observability** — LangSmith or Arize would be added for tracing, logging, and monitoring retrieval and generation quality in production.

---

## Related Project

This repo is the foundation for [agentic-rag](https://github.com/yourusername/agentic-rag), which extends this pipeline with tool use, query routing, multi-step reasoning, and agent memory.
