"""
store.py
--------
Responsible for creating, persisting, and loading the Chroma vector store.

Strategy: Load if exists, create if not.
    - First run: builds the index from embedded nodes and persists to disk
    - Subsequent runs: loads the existing index instantly without re-embedding
    - Force rebuild: set FORCE_REBUILD=true in .env or pass rebuild=True

Configuration (in order of precedence):
    1. Environment variables (CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, FORCE_REBUILD)
    2. Sensible defaults (persist_dir: ./data/chroma, collection: meridian_hr)

Production note:
    Chroma is used here for its zero-infrastructure local setup, making it
    ideal for development and portfolio demonstration. In a production system
    this would be replaced by a managed service such as Pinecone or Qdrant
    for scalability, persistence across deployments, and multi-tenant support.

Usage:
    from src.vectorstore.store import get_vector_store, build_index, load_index

    # First run — build and persist
    index = build_index(nodes)

    # Subsequent runs — load existing
    index = load_index()

    # Auto — load if exists, build if not
    index = get_vector_store(nodes)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# CHROMA_PERSIST_DIR: where Chroma stores its data on disk
# CHROMA_COLLECTION_NAME: logical name for the document collection
# FORCE_REBUILD: if "true", always rebuild the index from scratch
# -------------------------------------------------------------------
DEFAULT_PERSIST_DIR = "./data/chroma"
DEFAULT_COLLECTION_NAME = "meridian_hr"


def _get_config() -> tuple[str, str, bool]:
    """
    Resolve vector store configuration from env vars and defaults.

    Returns:
        Tuple of (persist_dir, collection_name, force_rebuild).
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() == "true"
    return persist_dir, collection_name, force_rebuild


def _get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    Initialize a persistent Chroma client.

    Args:
        persist_dir (str): Directory where Chroma persists data.

    Returns:
        chromadb.PersistentClient: Configured Chroma client.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Chroma persist directory: {persist_dir}")
    return chromadb.PersistentClient(path=persist_dir)


def _index_exists(persist_dir: str, collection_name: str) -> bool:
    """
    Check whether a persisted Chroma index already exists.

    Args:
        persist_dir (str): Path to the Chroma persist directory.
        collection_name (str): Name of the Chroma collection.

    Returns:
        bool: True if the index exists and contains data, False otherwise.
    """
    chroma_path = Path(persist_dir)
    if not chroma_path.exists():
        return False

    try:
        client = _get_chroma_client(persist_dir)
        collection = client.get_collection(collection_name)
        count = collection.count()
        logger.info(f"Existing index found: '{collection_name}' with {count} vectors")
        return count > 0
    except Exception:
        return False


def build_index(
    nodes: List[BaseNode],
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Build a new Chroma vector store index from embedded nodes and persist it.

    This should be called on first run or when a forced rebuild is needed.
    Embeddings must already be attached to the nodes (via embedder.py).

    Args:
        nodes (List[BaseNode]): Embedded document nodes from embedder.py.
        persist_dir (Optional[str]): Override for persist directory.
        collection_name (Optional[str]): Override for collection name.

    Returns:
        VectorStoreIndex: Built and persisted LlamaIndex vector store index.

    Raises:
        ValueError: If no nodes are provided.
    """
    if not nodes:
        raise ValueError("No nodes provided to build index.")

    cfg_persist_dir, cfg_collection, _ = _get_config()
    persist_dir = persist_dir or cfg_persist_dir
    collection_name = collection_name or cfg_collection

    logger.info(
        f"Building new index — collection: '{collection_name}' | "
        f"nodes: {len(nodes)}"
    )

    client = _get_chroma_client(persist_dir)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: '{collection_name}'")
    except Exception:
        pass

    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info(f"Index built and persisted to: {persist_dir}")
    return index


def load_index(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Load an existing Chroma vector store index from disk.

    Args:
        persist_dir (Optional[str]): Override for persist directory.
        collection_name (Optional[str]): Override for collection name.

    Returns:
        VectorStoreIndex: Loaded LlamaIndex vector store index.

    Raises:
        FileNotFoundError: If no persisted index is found at the given path.
    """
    cfg_persist_dir, cfg_collection, _ = _get_config()
    persist_dir = persist_dir or cfg_persist_dir
    collection_name = collection_name or cfg_collection

    if not _index_exists(persist_dir, collection_name):
        raise FileNotFoundError(
            f"No existing index found at '{persist_dir}' "
            f"for collection '{collection_name}'. "
            f"Run the pipeline first to build the index."
        )

    logger.info(f"Loading existing index — collection: '{collection_name}'")

    client = _get_chroma_client(persist_dir)
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    logger.info("Index loaded successfully")
    return index


def get_vector_store(
    nodes: Optional[List[BaseNode]] = None,
    rebuild: bool = False,
) -> VectorStoreIndex:
    """
    Main entry point — load index if it exists, build if it does not.

    This is the function called by pipeline.py. It handles the
    load-or-create logic automatically so the pipeline does not need
    to make that decision itself.

    Args:
        nodes (Optional[List[BaseNode]]): Embedded nodes required if
            building a new index. Not needed if loading an existing one.
        rebuild (bool): Force rebuild even if an index already exists.
            Can also be set via FORCE_REBUILD=true in .env.

    Returns:
        VectorStoreIndex: Ready-to-use vector store index.

    Raises:
        ValueError: If index does not exist and no nodes are provided.
    """
    persist_dir, collection_name, force_rebuild = _get_config()
    should_rebuild = rebuild or force_rebuild

    if not should_rebuild and _index_exists(persist_dir, collection_name):
        logger.info("Existing index detected — loading from disk")
        return load_index(persist_dir, collection_name)

    if not nodes:
        raise ValueError(
            "No existing index found and no nodes provided to build one. "
            "Run the full ingestion pipeline first or provide embedded nodes."
        )

    logger.info("No existing index found — building new index")
    return build_index(nodes, persist_dir, collection_name)


# -------------------------------------------------------------------
# Quick test — run directly to verify store works end to end
# python -m src.vectorstore.store
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model, embed_nodes

    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        docs = load_documents(test_path)
        nodes = chunk_documents(docs)
        embed_model = get_embed_model()
        embedded_nodes = embed_nodes(nodes, embed_model)

        index = get_vector_store(nodes=embedded_nodes)

        print(f"\n--- Store Test Results ---")
        print(f"Index type  : {type(index).__name__}")
        print(f"Persist dir : {os.getenv('CHROMA_PERSIST_DIR', DEFAULT_PERSIST_DIR)}")
        print(f"Collection  : {os.getenv('CHROMA_COLLECTION_NAME', DEFAULT_COLLECTION_NAME)}")
        print(f"\nIndex is ready for retrieval.")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
