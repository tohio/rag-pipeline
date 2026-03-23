"""
pipeline.py
-----------
Orchestrates the full end-to-end RAG pipeline.

This is the main entry point for the rag-pipeline project. It wires
together all components in the correct order:

    1. Load       — ingest documents from disk (loader.py)
    2. Chunk      — split documents into nodes (chunker.py)
    3. Embed      — generate embeddings for nodes (embedder.py)
    4. Index      — store nodes in Chroma vector store (store.py)
    5. Retrieve   — fetch relevant chunks for a query (retriever.py)
    6. Generate   — produce a grounded answer from context (generator.py)

The pipeline supports two modes:
    - Interactive mode: prompts the user for queries in a loop
    - Single query mode: accepts a query via CLI argument

Configuration:
    All component-level settings (chunk size, top-k, LLM provider, etc.)
    are managed via .env — see .env.example for all available options.

Usage:
    # Interactive mode
    python src/pipeline.py

    # Single query
    python src/pipeline.py --query "How many PTO days do I get?"

    # Force rebuild index
    python src/pipeline.py --rebuild

    # Custom document path
    python src/pipeline.py --docs data/raw/meridian-capital-handbook.pdf
"""

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import get_embed_model, embed_nodes
from src.vectorstore.store import get_vector_store
from src.retrieval.retriever import get_retriever, retrieve, format_retrieved_context
from src.generation.generator import get_llm, generate, build_response


from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Default document path
# -------------------------------------------------------------------
DEFAULT_DOCS_PATH = "data/raw"


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Encapsulates all components and manages initialization state.
    On first instantiation, builds the index from documents.
    On subsequent instantiations (index already on disk), loads
    the existing index without re-embedding.

    Attributes:
        docs_path (str): Path to source documents.
        rebuild (bool): Whether to force rebuild the index.
        retriever: Initialized vector index retriever.
        llm: Initialized language model.
    """

    def __init__(
        self,
        docs_path: str = DEFAULT_DOCS_PATH,
        rebuild: bool = False,
    ):
        self.docs_path = docs_path
        self.rebuild = rebuild
        self.retriever = None
        self.llm = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize all pipeline components.

        Handles the load-or-build decision for the vector index.
        If the index already exists and rebuild=False, skips ingestion
        and embedding entirely for a fast startup.
        """
        logger.info("Initializing RAG pipeline...")

        # Always initialize the LLM and embed model
        self.embed_model = get_embed_model()
        self.llm = get_llm()

        # Check if we need to build the index
        from src.vectorstore.store import _index_exists, _get_config
        persist_dir, collection_name, force_rebuild = _get_config()
        should_rebuild = self.rebuild or force_rebuild

        if not should_rebuild and _index_exists(persist_dir, collection_name):
            logger.info("Existing index found — skipping ingestion and embedding")
            index = get_vector_store()
        else:
            logger.info("Building index from documents...")
            docs = load_documents(self.docs_path)
            nodes = chunk_documents(docs)
            embedded_nodes = embed_nodes(nodes, self.embed_model)
            index = get_vector_store(nodes=embedded_nodes, rebuild=should_rebuild)

        self.retriever = get_retriever(index)
        logger.info("RAG pipeline ready")

    def query(self, question: str) -> dict:
        """
        Run a single query through the full RAG pipeline.

        Args:
            question (str): Natural language question from the user.

        Returns:
            dict: Structured response with answer and source citations.
                  Keys: query, answer, sources, num_sources
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info(f"Processing query: '{question}'")

        # Step 1: Retrieve relevant chunks
        results = retrieve(self.retriever, question)

        if not results:
            return {
                "query": question,
                "answer": (
                    "I couldn't find relevant information in the provided documents. "
                    "Please contact HR at hr@meridiancapitalgroup.com for assistance."
                ),
                "sources": [],
                "num_sources": 0,
            }

        # Step 2: Format context for the LLM
        context = format_retrieved_context(results)

        # Step 3: Generate answer
        answer = generate(self.llm, question, context)

        # Step 4: Build structured response
        response = build_response(question, answer, results)

        return response

    def print_response(self, response: dict) -> None:
        """
        Pretty print a pipeline response to the console.

        Args:
            response (dict): Response dict from query().
        """
        print("\n" + "=" * 60)
        print(f"QUESTION : {response['query']}")
        print("=" * 60)
        print(f"ANSWER   : {response['answer']}")
        print("-" * 60)
        print(f"SOURCES  : {response['num_sources']} chunk(s) retrieved")
        for i, src in enumerate(response["sources"], 1):
            print(
                f"  [{i}] {src['file_name']} | "
                f"Page: {src['page_label']} | "
                f"Score: {src['similarity_score']}"
            )
        print("=" * 60 + "\n")


def run_interactive(pipeline: RAGPipeline) -> None:
    """
    Run the pipeline in interactive mode.

    Prompts the user for queries in a loop until they type
    'exit', 'quit', or press Ctrl+C.

    Args:
        pipeline (RAGPipeline): Initialized pipeline instance.
    """
    print("\n" + "=" * 60)
    print(" Meridian Capital Group — HR Policy Assistant")
    print(" Powered by RAG | Type 'exit' to quit")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Ask a question: ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break

            response = pipeline.query(question)
            pipeline.print_response(response)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except ValueError as e:
            print(f"Input error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"Something went wrong: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Meridian Capital Group HR Policy RAG Pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=DEFAULT_DOCS_PATH,
        help=f"Path to source documents (default: {DEFAULT_DOCS_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the vector index from scratch",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    try:
        pipeline = RAGPipeline(
            docs_path=args.docs,
            rebuild=args.rebuild,
        )

        if args.query:
            # Single query mode
            response = pipeline.query(args.query)
            pipeline.print_response(response)
        else:
            # Interactive mode
            run_interactive(pipeline)

    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nConfiguration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nFile not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline error: {e}")
        sys.exit(1)
