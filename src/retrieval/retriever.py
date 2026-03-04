"""
retriever.py
------------
Responsible for querying the vector store and retrieving the most
relevant document chunks for a given user query.

Strategy: Dense vector retrieval using cosine similarity.
    - Converts the query to an embedding using the same model used
      during indexing (text-embedding-3-small)
    - Returns the top-k most similar nodes from the Chroma index

Configuration (in order of precedence):
    1. Function arguments
    2. Environment variables (RETRIEVAL_TOP_K, SIMILARITY_CUTOFF)
    3. Sensible defaults (top_k=5, similarity_cutoff=0.3)

Design note:
    The retriever is intentionally kept separate from the generator.
    This makes it easy to swap retrieval strategies (e.g. hybrid search,
    reranking, MMR) without touching generation logic, and allows the
    retrieval step to be evaluated independently.

Production note:
    Dense retrieval works well for semantic queries but can miss exact
    keyword matches. In production, a hybrid retrieval approach combining
    dense vectors with BM25 keyword search (available in Weaviate and
    Pinecone) typically outperforms either method alone.

Usage:
    from src.retrieval.retriever import get_retriever, retrieve

    retriever = get_retriever(index)
    results = retrieve(retriever, "What is the PTO policy?")
"""

import logging
import os
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# RETRIEVAL_TOP_K: number of chunks to retrieve per query
#   - 5 is a solid default — enough context without overwhelming the LLM
#   - increase for complex multi-part questions
#   - decrease for faster, more focused retrieval
#
# SIMILARITY_CUTOFF: minimum similarity score to include a result
#   - 0.3 filters out clearly irrelevant chunks
#   - lower = more results, potentially noisier
#   - higher = fewer results, potentially missing relevant context
# -------------------------------------------------------------------
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_CUTOFF = 0.3


def _get_config(
    top_k: Optional[int],
    similarity_cutoff: Optional[float],
) -> tuple[int, float]:
    """
    Resolve retrieval config from args > env vars > defaults.

    Args:
        top_k: Directly passed top-k value.
        similarity_cutoff: Directly passed similarity cutoff.

    Returns:
        Tuple of (top_k, similarity_cutoff).
    """
    resolved_top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K))
    resolved_cutoff = similarity_cutoff or float(
        os.getenv("SIMILARITY_CUTOFF", DEFAULT_SIMILARITY_CUTOFF)
    )
    return resolved_top_k, resolved_cutoff


def get_retriever(
    index: VectorStoreIndex,
    top_k: Optional[int] = None,
    similarity_cutoff: Optional[float] = None,
) -> VectorIndexRetriever:
    """
    Initialize and return a configured vector index retriever.

    Args:
        index (VectorStoreIndex): Built or loaded vector store index.
        top_k (Optional[int]): Number of chunks to retrieve. Defaults
            to RETRIEVAL_TOP_K env var or 5.
        similarity_cutoff (Optional[float]): Minimum similarity score.
            Defaults to SIMILARITY_CUTOFF env var or 0.3.

    Returns:
        VectorIndexRetriever: Configured retriever ready for querying.
    """
    resolved_top_k, resolved_cutoff = _get_config(top_k, similarity_cutoff)

    logger.info(
        f"Initializing retriever — "
        f"top_k={resolved_top_k} | similarity_cutoff={resolved_cutoff}"
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=resolved_top_k,
    )

    logger.info("Retriever ready")
    return retriever


def retrieve(
    retriever: VectorIndexRetriever,
    query: str,
    similarity_cutoff: Optional[float] = None,
) -> List[NodeWithScore]:
    """
    Retrieve the most relevant document chunks for a given query.

    Converts the query to an embedding, performs cosine similarity
    search against the vector store, and returns the top-k results
    above the similarity cutoff threshold.

    Args:
        retriever (VectorIndexRetriever): Initialized retriever.
        query (str): The user's natural language query.
        similarity_cutoff (Optional[float]): Filter out results below
            this similarity score. Defaults to SIMILARITY_CUTOFF env
            var or 0.3.

    Returns:
        List[NodeWithScore]: Retrieved chunks with similarity scores,
            sorted by relevance descending.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    _, resolved_cutoff = _get_config(None, similarity_cutoff)

    logger.info(f"Retrieving chunks for query: '{query}'")

    results = retriever.retrieve(query)

    # Filter by similarity cutoff
    filtered = [r for r in results if r.score >= resolved_cutoff]

    logger.info(
        f"Retrieved {len(results)} chunk(s) | "
        f"After cutoff filter: {len(filtered)} chunk(s)"
    )

    if not filtered:
        logger.warning(
            f"No chunks met the similarity cutoff of {resolved_cutoff}. "
            f"Consider lowering SIMILARITY_CUTOFF or checking your index."
        )

    return filtered


def format_retrieved_context(results: List[NodeWithScore]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.

    Each chunk is labelled with its source file and similarity score
    so the generator can reference where the context came from.

    Args:
        results (List[NodeWithScore]): Retrieved chunks from retrieve().

    Returns:
        str: Formatted context string ready to inject into a prompt.
    """
    if not results:
        return "No relevant context found."

    context_parts = []
    for i, result in enumerate(results, 1):
        file_name = result.node.metadata.get("file_name", "unknown")
        page = result.node.metadata.get("page_label", "unknown")
        score = result.score

        context_parts.append(
            f"[Chunk {i} | Source: {file_name} | Page: {page} | Score: {score:.3f}]\n"
            f"{result.node.get_content()}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_retrieval_metadata(results: List[NodeWithScore]) -> List[dict]:
    """
    Extract metadata from retrieved results for logging and evaluation.

    Args:
        results (List[NodeWithScore]): Retrieved chunks.

    Returns:
        List[dict]: Metadata for each retrieved chunk.
    """
    return [
        {
            "chunk_index": i,
            "node_id": result.node.node_id,
            "file_name": result.node.metadata.get("file_name", "unknown"),
            "page_label": result.node.metadata.get("page_label", "unknown"),
            "similarity_score": round(result.score, 4),
            "text_preview": result.node.text[:120].replace("\n", " ") + "...",
        }
        for i, result in enumerate(results, 1)
    ]


# -------------------------------------------------------------------
# Quick test — run directly to verify retriever works end to end
# python -m src.retrieval.retriever
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model, embed_nodes
    from src.vectorstore.store import get_vector_store

    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "What is the PTO policy?"

    try:
        docs = load_documents(test_path)
        nodes = chunk_documents(docs)
        embed_model = get_embed_model()
        embedded_nodes = embed_nodes(nodes, embed_model)
        index = get_vector_store(nodes=embedded_nodes)

        retriever = get_retriever(index)
        results = retrieve(retriever, test_query)
        metadata = get_retrieval_metadata(results)

        print(f"\n--- Retriever Test Results ---")
        print(f"Query      : {test_query}")
        print(f"Chunks found: {len(results)}\n")
        for meta in metadata:
            print(f"  Chunk {meta['chunk_index']} | "
                  f"Score: {meta['similarity_score']} | "
                  f"Page: {meta['page_label']}")
            print(f"  Preview: {meta['text_preview']}\n")

        print(f"\n--- Formatted Context ---")
        print(format_retrieved_context(results))

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
