"""
chunker.py
----------
Responsible for splitting loaded documents into smaller chunks
suitable for embedding and retrieval.

Strategy: Fixed size token-based chunking with overlap.
    - Chunk size: number of tokens per chunk
    - Chunk overlap: number of tokens shared between adjacent chunks
      (preserves context across chunk boundaries)

Configuration (in order of precedence):
    1. Passed directly as arguments to chunk_documents()
    2. Environment variables (CHUNK_SIZE, CHUNK_OVERLAP)
    3. Sensible defaults (chunk_size=512, chunk_overlap=50)

Usage:
    from src.ingestion.chunker import chunk_documents
    nodes = chunk_documents(documents)

    # Override defaults
    nodes = chunk_documents(documents, chunk_size=256, chunk_overlap=32)
"""

import logging
import os
from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults — override via .env or function arguments
# CHUNK_SIZE: number of tokens per chunk
#   - 512 is a solid default for dense policy/HR documents
#   - smaller (256) = more precise retrieval, more chunks
#   - larger (1024) = more context per chunk, fewer chunks
#
# CHUNK_OVERLAP: tokens shared between adjacent chunks
#   - 50 preserves sentence continuity across boundaries
#   - ~10% of chunk size is a good rule of thumb
# -------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def _get_config(
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> tuple[int, int]:
    """
    Resolve chunk size and overlap from args > env vars > defaults.

    Args:
        chunk_size: Directly passed chunk size (highest priority).
        chunk_overlap: Directly passed chunk overlap (highest priority).

    Returns:
        Tuple of (chunk_size, chunk_overlap) as integers.
    """
    resolved_size = (
        chunk_size
        or int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    )
    resolved_overlap = (
        chunk_overlap
        or int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    )

    if resolved_overlap >= resolved_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({resolved_overlap}) must be less than "
            f"CHUNK_SIZE ({resolved_size})"
        )

    return resolved_size, resolved_overlap


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[BaseNode]:
    """
    Split a list of documents into smaller chunks (nodes).

    Uses LlamaIndex's SentenceSplitter which respects sentence
    boundaries while targeting the specified token chunk size.
    This avoids splitting mid-sentence, which improves chunk quality.

    Args:
        documents (List[Document]): Loaded documents from loader.py.
        chunk_size (Optional[int]): Tokens per chunk. Defaults to
            CHUNK_SIZE env var or 512.
        chunk_overlap (Optional[int]): Overlapping tokens between
            chunks. Defaults to CHUNK_OVERLAP env var or 50.

    Returns:
        List[BaseNode]: List of LlamaIndex nodes (chunks) ready
            for embedding.

    Raises:
        ValueError: If documents list is empty or overlap >= chunk size.
    """
    if not documents:
        raise ValueError("No documents provided to chunk.")

    resolved_size, resolved_overlap = _get_config(chunk_size, chunk_overlap)

    logger.info(
        f"Chunking {len(documents)} document(s) | "
        f"chunk_size={resolved_size} | chunk_overlap={resolved_overlap}"
    )

    # Configure the splitter
    splitter = SentenceSplitter(
        chunk_size=resolved_size,
        chunk_overlap=resolved_overlap,
    )

    # Parse documents into nodes
    nodes = splitter.get_nodes_from_documents(documents)

    logger.info(f"Produced {len(nodes)} chunk(s) from {len(documents)} document(s)")
    _log_chunk_stats(nodes)

    return nodes


def _log_chunk_stats(nodes: List[BaseNode]) -> None:
    """
    Log basic statistics about the produced chunks.
    Useful for verifying chunk quality before embedding.

    Args:
        nodes (List[BaseNode]): List of chunked nodes.
    """
    if not nodes:
        return

    lengths = [len(node.text) for node in nodes]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    logger.info(
        f"Chunk stats — "
        f"avg chars: {avg_len:.0f} | "
        f"min chars: {min_len} | "
        f"max chars: {max_len}"
    )


def get_chunk_metadata(nodes: List[BaseNode]) -> List[dict]:
    """
    Extract metadata from chunked nodes.
    Useful for debugging and inspection before indexing.

    Args:
        nodes (List[BaseNode]): List of LlamaIndex nodes.

    Returns:
        List[dict]: Metadata for each chunk including index,
            source file, and a text preview.
    """
    return [
        {
            "chunk_index": i,
            "node_id": node.node_id,
            "file_name": node.metadata.get("file_name", "unknown"),
            "page_label": node.metadata.get("page_label", "unknown"),
            "char_count": len(node.text),
            "text_preview": node.text[:120].replace("\n", " ") + "...",
        }
        for i, node in enumerate(nodes)
    ]


# -------------------------------------------------------------------
# Quick test — run directly to verify chunker works end to end
# python -m src.ingestion.chunker
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.ingestion.loader import load_documents

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        docs = load_documents(test_path)
        nodes = chunk_documents(docs)
        metadata = get_chunk_metadata(nodes)

        print(f"\n--- Chunker Test Results ---")
        print(f"Total chunks produced: {len(nodes)}")
        print(f"\nFirst 5 chunks:\n")
        for meta in metadata[:5]:
            print(f"  Chunk {meta['chunk_index']} | "
                  f"Page: {meta['page_label']} | "
                  f"Chars: {meta['char_count']}")
            print(f"  Preview: {meta['text_preview']}\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
