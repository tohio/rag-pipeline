"""
generator.py
------------
Responsible for generating answers from retrieved context using an LLM.

The generator takes the user query and the retrieved document chunks,
constructs a prompt that grounds the LLM in the retrieved context,
and returns a final answer.

Supported LLMs (configurable via LLM_PROVIDER in .env):
    - openai   : OpenAI GPT-4o (default)
    - anthropic: Anthropic Claude
    - (extensible — add new providers in _get_llm())

Configuration (in order of precedence):
    1. Environment variables (LLM_PROVIDER, OPENAI_API_KEY,
       ANTHROPIC_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS)
    2. Sensible defaults (provider: openai, temp: 0.1, max_tokens: 512)

Design note:
    Temperature is set low (0.1) by default. For a Q&A use case over
    policy documents, factual accuracy matters more than creativity.
    Higher temperatures increase the risk of the model embellishing
    or hallucinating beyond the retrieved context.

Production note:
    In production, the generator would stream responses token by token
    for a better user experience. LlamaIndex supports streaming via
    query_engine.aquery() with async generators. This is called out
    in the README as a production consideration.

Usage:
    from src.generation.generator import get_llm, generate

    llm = get_llm()
    response = generate(llm, query, context)
"""

import logging
import os
from typing import Optional

from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# LLM_PROVIDER   : which LLM provider to use (openai | anthropic)
# LLM_TEMPERATURE: controls randomness — low for factual Q&A
# LLM_MAX_TOKENS : max tokens in the generated response
# -------------------------------------------------------------------
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512

# -------------------------------------------------------------------
# Prompt template
# Instructs the LLM to answer strictly from retrieved context.
# The "If the context does not contain..." instruction is critical —
# it prevents the model from falling back on training data, which
# would defeat the purpose of RAG.
# -------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """\
You are a helpful HR assistant for Meridian Capital Group.
Answer the employee's question using ONLY the context provided below.
Be concise, accurate, and professional.

If the context does not contain enough information to answer the question,
say: "I don't have enough information in the provided documents to answer that.
Please contact HR at hr@meridiancapitalgroup.com for assistance."

Do not make up information or draw from knowledge outside the provided context.

---
CONTEXT:
{context}
---

QUESTION:
{query}

ANSWER:
"""


def get_llm(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    Initialize and return the configured LLM.

    Reads provider and model settings from environment variables.
    Supports OpenAI and Anthropic out of the box. New providers
    can be added by extending the if/elif block in this function.

    Args:
        provider (Optional[str]): LLM provider override ('openai' or
            'anthropic'). Defaults to LLM_PROVIDER env var or 'openai'.
        temperature (Optional[float]): Sampling temperature override.
            Defaults to LLM_TEMPERATURE env var or 0.1.
        max_tokens (Optional[int]): Max output tokens override.
            Defaults to LLM_MAX_TOKENS env var or 512.

    Returns:
        LLM: Configured LlamaIndex-compatible LLM instance.

    Raises:
        EnvironmentError: If the required API key is not set.
        ValueError: If an unsupported provider is specified.
    """
    resolved_provider = (
        provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    ).lower()

    resolved_temp = temperature or float(
        os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    resolved_max_tokens = max_tokens or int(
        os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )

    logger.info(
        f"Initializing LLM — provider: {resolved_provider} | "
        f"temperature: {resolved_temp} | max_tokens: {resolved_max_tokens}"
    )

    if resolved_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Please add it to your .env file."
            )
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL_OPENAI)
        llm = OpenAI(
            model=model,
            api_key=api_key,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
        )
        logger.info(f"OpenAI LLM ready: {model}")
        return llm

    elif resolved_provider == "anthropic":
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "llama-index-llms-anthropic is not installed. "
                "Run: pip install llama-index-llms-anthropic"
            )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file."
            )
        model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL_ANTHROPIC)
        llm = Anthropic(
            model=model,
            api_key=api_key,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
        )
        logger.info(f"Anthropic LLM ready: {model}")
        return llm

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            f"Supported providers: openai, anthropic"
        )


def generate(
    llm,
    query: str,
    context: str,
) -> str:
    """
    Generate an answer from retrieved context using the LLM.

    Constructs a grounded prompt using the RAG_PROMPT_TEMPLATE,
    injects the query and retrieved context, and returns the
    LLM's response as a string.

    Args:
        llm: Initialized LLM instance from get_llm().
        query (str): The user's natural language question.
        context (str): Formatted retrieved context from retriever.py.

    Returns:
        str: Generated answer grounded in the retrieved context.

    Raises:
        ValueError: If query or context is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        query=query.strip(),
    )

    logger.info(f"Generating answer for query: '{query}'")

    response = llm.complete(prompt)
    answer = response.text.strip()

    logger.info(f"Answer generated ({len(answer)} chars)")
    return answer


def build_response(
    query: str,
    answer: str,
    retrieved_chunks: list[NodeWithScore],
) -> dict:
    """
    Build a structured response object combining the answer and sources.

    This is what gets returned to the user and displayed in the UI.
    Including sources allows the user to verify the answer against
    the original document — a key trust signal in enterprise RAG.

    Args:
        query (str): Original user query.
        answer (str): Generated answer from generate().
        retrieved_chunks (list[NodeWithScore]): Chunks used as context.

    Returns:
        dict: Structured response with answer and source citations.
    """
    sources = [
        {
            "file_name": chunk.node.metadata.get("file_name", "unknown"),
            "page_label": chunk.node.metadata.get("page_label", "unknown"),
            "similarity_score": round(chunk.score, 4),
            "text_preview": chunk.node.text[:200].replace("\n", " ") + "...",
        }
        for chunk in retrieved_chunks
    ]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }


# -------------------------------------------------------------------
# Quick test — run directly to verify generator works end to end
# python -m src.generation.generator
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model, embed_nodes
    from src.vectorstore.store import get_vector_store
    from src.retrieval.retriever import get_retriever, retrieve, format_retrieved_context

    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "How many PTO days do I get in my first year?"

    try:
        docs = load_documents(test_path)
        nodes = chunk_documents(docs)
        embed_model = get_embed_model()
        embedded_nodes = embed_nodes(nodes, embed_model)
        index = get_vector_store(nodes=embedded_nodes)

        retriever = get_retriever(index)
        results = retrieve(retriever, test_query)
        context = format_retrieved_context(results)

        llm = get_llm()
        answer = generate(llm, test_query, context)
        response = build_response(test_query, answer, results)

        print(f"\n--- Generator Test Results ---")
        print(f"Query  : {response['query']}")
        print(f"Answer : {response['answer']}")
        print(f"\nSources ({response['num_sources']}):")
        for src in response["sources"]:
            print(f"  - {src['file_name']} | Page: {src['page_label']} | Score: {src['similarity_score']}")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
