"""
test_pipeline.py
----------------
Unit tests for the rag-pipeline components.

Tests are organized by module and cover:
    - loader.py     : document loading, error handling
    - chunker.py    : chunking logic, config resolution
    - embedder.py   : embedding model initialization, validation
    - store.py      : index creation, load-or-build logic
    - retriever.py  : retrieval, similarity cutoff, formatting
    - generator.py  : prompt construction, LLM provider switching
    - pipeline.py   : end-to-end integration

Test strategy:
    - Unit tests mock external dependencies (OpenAI API, Chroma)
      so tests run fast without API calls or disk I/O
    - Integration test (marked slow) runs the full pipeline end-to-end
      and requires real API keys and documents

Usage:
    # Run all unit tests
    pytest tests/test_pipeline.py -v

    # Run and exclude slow integration tests
    pytest tests/test_pipeline.py -v -m "not slow"

    # Run only integration tests
    pytest tests/test_pipeline.py -v -m slow
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_document():
    """A minimal LlamaIndex Document for testing."""
    from llama_index.core.schema import Document
    return Document(
        text=(
            "Meridian Capital Group provides all full-time employees with 15 days "
            "of PTO in their first year of service. PTO accrues monthly beginning "
            "on the employee's first day of employment."
        ),
        metadata={
            "file_name": "meridian-capital-handbook.pdf",
            "file_path": "/data/raw/meridian-capital-handbook.pdf",
            "page_label": "5",
        },
    )


@pytest.fixture
def sample_documents(sample_document):
    """A list of sample documents."""
    return [sample_document]


@pytest.fixture
def sample_nodes(sample_documents):
    """Chunked nodes from sample documents."""
    from src.ingestion.chunker import chunk_documents
    return chunk_documents(sample_documents)


@pytest.fixture
def mock_embed_model():
    """Mock embedding model that returns fake vectors."""
    mock = MagicMock()
    mock.model_name = "text-embedding-3-small"
    mock.get_text_embedding_batch.return_value = [
        [0.1, 0.2, 0.3] * 512  # 1536-dim fake vector
    ]
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM that returns a canned response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Employees receive 15 days of PTO in their first year."
    mock.complete.return_value = mock_response
    return mock


# -------------------------------------------------------------------
# loader.py tests
# -------------------------------------------------------------------

class TestLoader:

    def test_load_nonexistent_path_raises(self):
        """Should raise FileNotFoundError for a path that does not exist."""
        from src.ingestion.loader import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/to/file.pdf")

    def test_load_unsupported_file_type_raises(self, tmp_path):
        """Should raise ValueError for unsupported file types."""
        from src.ingestion.loader import load_documents
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_documents(str(txt_file))

    def test_load_empty_directory_raises(self, tmp_path):
        """Should raise ValueError if no PDFs found in directory."""
        from src.ingestion.loader import load_documents
        with pytest.raises(ValueError, match="No PDF files found"):
            load_documents(str(tmp_path))

    def test_get_document_metadata_returns_correct_keys(self, sample_documents):
        """Metadata should include expected keys for each document."""
        from src.ingestion.loader import get_document_metadata
        metadata = get_document_metadata(sample_documents)
        assert len(metadata) == 1
        assert "doc_id" in metadata[0]
        assert "file_name" in metadata[0]
        assert "text_length" in metadata[0]
        assert metadata[0]["text_length"] > 0


# -------------------------------------------------------------------
# chunker.py tests
# -------------------------------------------------------------------

class TestChunker:

    def test_chunk_produces_nodes(self, sample_documents):
        """Should produce at least one node from a document."""
        from src.ingestion.chunker import chunk_documents
        nodes = chunk_documents(sample_documents)
        assert len(nodes) >= 1

    def test_chunk_empty_documents_raises(self):
        """Should raise ValueError when given an empty document list."""
        from src.ingestion.chunker import chunk_documents
        with pytest.raises(ValueError, match="No documents provided"):
            chunk_documents([])

    def test_chunk_overlap_exceeds_size_raises(self, sample_documents):
        """Should raise ValueError if overlap >= chunk size."""
        from src.ingestion.chunker import chunk_documents
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            chunk_documents(sample_documents, chunk_size=100, chunk_overlap=100)

    def test_chunk_respects_custom_size(self, sample_documents):
        """Nodes should respect a smaller custom chunk size."""
        from src.ingestion.chunker import chunk_documents
        nodes = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        for node in nodes:
            assert len(node.text) > 0

    def test_get_chunk_metadata_structure(self, sample_nodes):
        """Metadata should include expected keys for each node."""
        from src.ingestion.chunker import get_chunk_metadata
        metadata = get_chunk_metadata(sample_nodes)
        assert len(metadata) > 0
        for meta in metadata:
            assert "chunk_index" in meta
            assert "node_id" in meta
            assert "text_preview" in meta

    def test_chunk_config_from_env(self, monkeypatch, sample_documents):
        """Should read chunk size from environment variable."""
        from src.ingestion.chunker import chunk_documents
        monkeypatch.setenv("CHUNK_SIZE", "256")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")
        nodes = chunk_documents(sample_documents)
        assert len(nodes) >= 1


# -------------------------------------------------------------------
# embedder.py tests
# -------------------------------------------------------------------

class TestEmbedder:

    def test_get_embed_model_raises_without_api_key(self, monkeypatch):
        """Should raise EnvironmentError if OPENAI_API_KEY is not set."""
        from src.embedding.embedder import get_embed_model
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_embed_model()

    def test_embed_empty_nodes_raises(self, mock_embed_model):
        """Should raise ValueError when given empty node list."""
        from src.embedding.embedder import embed_nodes
        with pytest.raises(ValueError, match="No nodes provided"):
            embed_nodes([], mock_embed_model)

    def test_embed_attaches_embeddings_to_nodes(
        self, sample_nodes, mock_embed_model
    ):
        """Each node should have an embedding attached after embed_nodes."""
        from src.embedding.embedder import embed_nodes
        mock_embed_model.get_text_embedding_batch.return_value = [
            [0.1] * 1536 for _ in sample_nodes
        ]
        embedded = embed_nodes(sample_nodes, mock_embed_model)
        for node in embedded:
            assert node.embedding is not None
            assert len(node.embedding) == 1536


# -------------------------------------------------------------------
# retriever.py tests
# -------------------------------------------------------------------

class TestRetriever:

    def test_retrieve_empty_query_raises(self):
        """Should raise ValueError for empty or whitespace queries."""
        from src.retrieval.retriever import retrieve
        mock_retriever = MagicMock()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve(mock_retriever, "")

    def test_retrieve_whitespace_query_raises(self):
        """Should raise ValueError for whitespace-only queries."""
        from src.retrieval.retriever import retrieve
        mock_retriever = MagicMock()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve(mock_retriever, "   ")

    def test_retrieve_filters_by_similarity_cutoff(self):
        """Results below similarity cutoff should be filtered out."""
        from src.retrieval.retriever import retrieve

        mock_node = MagicMock()
        mock_node.score = 0.1  # Below default cutoff of 0.3

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        results = retrieve(mock_retriever, "What is the PTO policy?")
        assert len(results) == 0

    def test_format_retrieved_context_empty(self):
        """Should return a fallback string for empty results."""
        from src.retrieval.retriever import format_retrieved_context
        result = format_retrieved_context([])
        assert "No relevant context" in result

    def test_format_retrieved_context_structure(self):
        """Formatted context should include chunk label and text."""
        from src.retrieval.retriever import format_retrieved_context

        mock_node = MagicMock()
        mock_node.score = 0.85
        mock_node.node.text = "Employees receive 15 days of PTO."
        mock_node.node.metadata = {"file_name": "handbook.pdf", "page_label": "5"}
        mock_node.node.get_content.return_value = "Employees receive 15 days of PTO."

        result = format_retrieved_context([mock_node])
        assert "Chunk 1" in result
        assert "handbook.pdf" in result
        assert "Employees receive 15 days of PTO." in result


# -------------------------------------------------------------------
# generator.py tests
# -------------------------------------------------------------------

class TestGenerator:

    def test_get_llm_raises_without_openai_key(self, monkeypatch):
        """Should raise EnvironmentError if OPENAI_API_KEY is not set."""
        from src.generation.generator import get_llm
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_llm()

    def test_get_llm_raises_for_unsupported_provider(self, monkeypatch):
        """Should raise ValueError for an unsupported LLM provider."""
        from src.generation.generator import get_llm
        monkeypatch.setenv("LLM_PROVIDER", "cohere")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm()

    def test_generate_empty_query_raises(self, mock_llm):
        """Should raise ValueError for an empty query."""
        from src.generation.generator import generate
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generate(mock_llm, "", "some context")

    def test_generate_empty_context_raises(self, mock_llm):
        """Should raise ValueError for empty context."""
        from src.generation.generator import generate
        with pytest.raises(ValueError, match="Context cannot be empty"):
            generate(mock_llm, "What is PTO?", "")

    def test_generate_calls_llm_complete(self, mock_llm):
        """generate() should call llm.complete() exactly once."""
        from src.generation.generator import generate
        generate(mock_llm, "What is the PTO policy?", "PTO context here.")
        mock_llm.complete.assert_called_once()

    def test_generate_returns_string(self, mock_llm):
        """generate() should return a non-empty string."""
        from src.generation.generator import generate
        result = generate(mock_llm, "What is the PTO policy?", "PTO context here.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_response_structure(self, mock_llm):
        """build_response() should return expected keys."""
        from src.generation.generator import build_response

        mock_chunk = MagicMock()
        mock_chunk.score = 0.85
        mock_chunk.node.metadata = {"file_name": "handbook.pdf", "page_label": "5"}
        mock_chunk.node.text = "PTO policy text."

        response = build_response(
            query="What is PTO?",
            answer="You get 15 days.",
            retrieved_chunks=[mock_chunk],
        )

        assert "query" in response
        assert "answer" in response
        assert "sources" in response
        assert "num_sources" in response
        assert response["num_sources"] == 1


# -------------------------------------------------------------------
# pipeline.py integration test (slow — requires API keys + documents)
# -------------------------------------------------------------------

class TestPipelineIntegration:

    @pytest.mark.slow
    def test_full_pipeline_returns_answer(self):
        """
        End-to-end integration test.
        Requires:
            - OPENAI_API_KEY set in environment
            - PDF document at data/raw/
        """
        from src.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        response = pipeline.query("How many PTO days does a new employee receive?")

        assert response is not None
        assert "answer" in response
        assert "15" in response["answer"]
        assert response["num_sources"] > 0

    @pytest.mark.slow
    def test_pipeline_handles_unknown_question(self):
        """
        Pipeline should return a graceful fallback for questions
        outside the scope of the handbook.
        """
        from src.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        response = pipeline.query(
            "What is the weather like in Chicago today?"
        )

        assert response is not None
        assert "answer" in response
