"""
embedder.py
-----------
Responsible for initializing and configuring the embedding model
used to convert document chunks into vector representations.

Model: OpenAI text-embedding-3-small
    - 1536 dimensions
    - Fast, cost-effective, strong performance for retrieval tasks
    - Good default for enterprise document Q&A use cases

Configuration (in order of precedence):
    1. Environment variables (OPENAI_API_KEY, EMBEDDING_MODEL)
    2. Sensible defaults (model: text-embedding-3-small)

The embedder is designed to be provider-agnostic at the interface level.
Swapping to text-embedding-3-large or a HuggingFace model only requires
changing the returned embed_model object — nothing downstream changes.

Production note:
    For high-volume production workloads, consider text-embedding-3-large
    for higher accuracy or a locally hosted HuggingFace model to eliminate
    per-token API costs entirely.

Usage:
    from src.embedding.embedder import get_embed_model, embed_nodes
    embed_model = get_embed_model()
    nodes = embed_nodes(nodes, embed_model)
"""

import logging
import os
from typing import List

from llama_index.core import Settings
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# EMBEDDING_MODEL: OpenAI embedding model name
#   - text-embedding-3-small: fast, cheap, 1536 dims — good default
#   - text-embedding-3-large: higher accuracy, more expensive, 3072 dims
# -------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embed_model() -> OpenAIEmbedding:
    """
    Initialize and return the OpenAI embedding model.

    Reads configuration from environment variables with sensible defaults.
    Registers the model globally in LlamaIndex Settings so it is
    automatically used by the vector store and retriever downstream.

    Returns:
        OpenAIEmbedding: Configured embedding model instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Please add it to your .env file or environment variables."
        )

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    logger.info(f"Initializing embedding model: {model_name}")

    embed_model = OpenAIEmbedding(
        model=model_name,
        api_key=api_key,
    )

    # Register globally so LlamaIndex components use it automatically
    Settings.embed_model = embed_model

    logger.info(f"Embedding model ready: {model_name}")
    return embed_model


def embed_nodes(
    nodes: List[BaseNode],
    embed_model: OpenAIEmbedding,
) -> List[BaseNode]:
    """
    Generate embeddings for a list of document nodes.

    Embeddings are attached directly to each node object and are
    persisted when the nodes are indexed into the vector store.

    Args:
        nodes (List[BaseNode]): Chunked document nodes from chunker.py.
        embed_model (OpenAIEmbedding): Initialized embedding model.

    Returns:
        List[BaseNode]: Same nodes with embeddings attached.

    Raises:
        ValueError: If nodes list is empty.
    """
    if not nodes:
        raise ValueError("No nodes provided for embedding.")

    logger.info(f"Embedding {len(nodes)} node(s) using {embed_model.model_name}...")

    # Extract text from each node for batch embedding
    texts = [node.get_content() for node in nodes]

    # Batch embed for efficiency — minimizes API round trips
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    # Attach embeddings back to nodes
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    logger.info(f"Successfully embedded {len(nodes)} node(s)")
    _log_embedding_stats(nodes)

    return nodes


def _log_embedding_stats(nodes: List[BaseNode]) -> None:
    """
    Log basic embedding statistics for verification.

    Args:
        nodes (List[BaseNode]): Nodes with embeddings attached.
    """
    if not nodes or nodes[0].embedding is None:
        return

    dims = len(nodes[0].embedding)
    logger.info(
        f"Embedding stats — "
        f"total nodes: {len(nodes)} | "
        f"embedding dimensions: {dims}"
    )


# -------------------------------------------------------------------
# Quick test — run directly to verify embedder works end to end
# python -m src.embedding.embedder
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents

    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        docs = load_documents(test_path)
        nodes = chunk_documents(docs)

        embed_model = get_embed_model()
        embedded_nodes = embed_nodes(nodes, embed_model)

        print(f"\n--- Embedder Test Results ---")
        print(f"Total nodes embedded: {len(embedded_nodes)}")
        print(f"Embedding dimensions: {len(embedded_nodes[0].embedding)}")
        print(f"\nSample node:")
        print(f"  Node ID   : {embedded_nodes[0].node_id}")
        print(f"  Text preview: {embedded_nodes[0].text[:120].replace(chr(10), ' ')}...")
        print(f"  Embedding (first 5 dims): {embedded_nodes[0].embedding[:5]}")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
