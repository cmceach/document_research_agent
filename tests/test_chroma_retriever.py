"""
Unit tests for ChromaRetriever
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from src.retriever.chroma_retriever import ChromaRetriever

# Mock environment variables
@pytest.fixture
def mock_env_vars():
    # Save original environment variables
    orig_vars = {
        "CHROMA_DB_PATH": os.environ.get("CHROMA_DB_PATH"),
        "CHROMA_COLLECTION_NAME": os.environ.get("CHROMA_COLLECTION_NAME"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_EMBEDDING_MODEL_NAME": os.environ.get("OPENAI_EMBEDDING_MODEL_NAME")
    }
    
    # Set test environment variables
    os.environ["CHROMA_DB_PATH"] = "./test_chroma_db"
    os.environ["CHROMA_COLLECTION_NAME"] = "test_collection"
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_EMBEDDING_MODEL_NAME"] = "text-embedding-ada-002"
    
    yield
    
    # Restore original environment variables
    for key, value in orig_vars.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

@pytest.fixture
def mock_chroma_client():
    # Create a mock ChromaDB client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_collection.count.return_value = 10
    return mock_client, mock_collection

@pytest.fixture
def mock_openai_client():
    # Create a mock OpenAI client
    mock_client = MagicMock()
    mock_embedding_response = MagicMock()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3]
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response
    return mock_client

# Test initialization
@patch('chromadb.PersistentClient')
@patch('openai.OpenAI')
@patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
def test_init(mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
    # Setup mocks
    mock_chroma.return_value = mock_chroma_client[0]
    mock_openai.return_value = mock_openai_client
    mock_embedding_func.return_value = MagicMock()
    
    # Create retriever instance with lazy_init=False to force initialization
    retriever = ChromaRetriever(lazy_init=False)
    
    # Only assert the attribute values
    assert retriever.chroma_db_path == "./test_chroma_db"
    assert retriever.collection_name == "test_collection"
    assert retriever.openai_api_key == "test_api_key"
    assert retriever.embedding_model == "text-embedding-ada-002"

# Test retrieve_context method
@patch('chromadb.PersistentClient')
@patch('openai.OpenAI')
@patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
def test_retrieve_context(mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
    # Setup mocks
    mock_chroma.return_value = mock_chroma_client[0]
    mock_collection = mock_chroma_client[1]
    mock_openai.return_value = mock_openai_client
    mock_embedding_func.return_value = MagicMock()
    
    # Setup mock query response
    mock_query_response = {
        "documents": [["This is document 1", "This is document 2"]],
        "metadatas": [[
            {"filename": "doc1.pdf", "page_number": 1},
            {"filename": "doc2.pdf", "page_number": 2}
        ]]
    }
    mock_collection.query.return_value = mock_query_response
    
    # Create retriever instance and call retrieve_context
    retriever = ChromaRetriever()
    results = retriever.retrieve_context(
        search_queries=["test query"],
        filenames=["doc1.pdf", "doc2.pdf"],
        top_k=2
    )
    
    # Assert retrieve_context results
    assert len(results) == 2
    assert results[0]["text"] == "This is document 1"
    assert results[0]["page"] == 1
    assert results[0]["filename"] == "doc1.pdf"
    assert results[1]["text"] == "This is document 2"
    assert results[1]["page"] == 2
    assert results[1]["filename"] == "doc2.pdf"
    
    # Assert query was called with correct parameters
    mock_collection.query.assert_called_once_with(
        query_texts=["test query"],
        n_results=2,
        where={"filename": {"$in": ["doc1.pdf", "doc2.pdf"]}},
        include=["documents", "metadatas"]
    )

# Test get_collection_stats method
@patch('chromadb.PersistentClient')
@patch('openai.OpenAI')
@patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
def test_get_collection_stats(mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
    # Setup mocks
    mock_chroma.return_value = mock_chroma_client[0]
    mock_collection = mock_chroma_client[1]
    mock_openai.return_value = mock_openai_client
    mock_embedding_func.return_value = MagicMock()
    mock_collection.count.return_value = 42
    
    # Create retriever instance and call get_collection_stats
    retriever = ChromaRetriever()
    stats = retriever.get_collection_stats()
    
    # Assert stats results
    assert stats["collection_name"] == "test_collection"
    assert stats["document_count"] == 42
    assert stats["embedding_model"] == "text-embedding-ada-002" 