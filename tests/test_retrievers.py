import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retriever.base_retriever import BaseRetriever
from src.retriever.chroma_retriever import ChromaRetriever
from src.retriever.azure_search_retriever import AzureSearchRetriever
import os

class TestBaseRetriever:
    """Tests for the BaseRetriever class."""
    
    class ConcreteRetriever(BaseRetriever):
        """Concrete implementation for testing abstract base class."""
        def retrieve_context(self, search_queries, filenames, top_k=5):
            return []
    
    def test_generate_embeddings_caching(self):
        """Test that embeddings are properly cached."""
        retriever = self.ConcreteRetriever()
        retriever.openai_api_key = "test_key"
        retriever.embedding_model = "test_model"
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        retriever.openai_client = mock_client
        
        # First call should use the client
        result1 = retriever.generate_embeddings("test text")
        assert result1 == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()
        
        # Second call should use cache
        mock_client.embeddings.create.reset_mock()
        result2 = retriever.generate_embeddings("test text")
        assert result2 == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_not_called()

class TestChromaRetriever:
    """Tests for the ChromaRetriever class."""
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up environment variables for testing."""
        monkeypatch.setenv("CHROMA_DB_PATH", "./test_db")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

    @pytest.fixture
    def mock_env_vars(self):
        """Alternative environment setup for compatibility."""
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
    def mock_chroma_client(self):
        """Create a mock ChromaDB client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 10
        return mock_client, mock_collection

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_embedding_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_embedding_response
        return mock_client
    
    def test_batch_retrieve_context(self, mock_env):
        """Test batch retrieval functionality."""
        retriever = ChromaRetriever(lazy_init=True)
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"], ["doc3", "doc4"]],
            "metadatas": [[
                {"filename": "test1.txt", "page_number": 1},
                {"filename": "test2.txt", "page_number": 2}
            ], [
                {"filename": "test3.txt", "page_number": 3},
                {"filename": "test4.txt", "page_number": 4}
            ]]
        }
        retriever.collection = mock_collection
        
        # Test batch retrieval
        results = retriever.batch_retrieve_context(
            search_queries=["query1", "query2"],
            filenames=["test1.txt", "test2.txt"],
            top_k=2
        )
        
        assert len(results) == 4  # Should get 4 unique results
        assert all(isinstance(r, dict) for r in results)
        assert all(set(r.keys()) == {"text", "page", "filename"} for r in results)

    @patch('chromadb.PersistentClient')
    @patch('openai.OpenAI')
    @patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
    def test_init(self, mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
        """Test ChromaRetriever initialization."""
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

    @patch('chromadb.PersistentClient')
    @patch('openai.OpenAI')
    @patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
    def test_retrieve_context(self, mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
        """Test ChromaRetriever context retrieval."""
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
            include=["documents", "metadatas", "distances"]
        )

    @patch('chromadb.PersistentClient')
    @patch('openai.OpenAI')
    @patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction')
    def test_get_collection_stats(self, mock_embedding_func, mock_openai, mock_chroma, mock_env_vars, mock_chroma_client, mock_openai_client):
        """Test ChromaRetriever collection statistics."""
        # Setup mocks
        mock_chroma.return_value = mock_chroma_client[0]
        mock_collection = mock_chroma_client[1]
        mock_openai.return_value = mock_openai_client
        mock_embedding_func.return_value = MagicMock()
        mock_collection.count.return_value = 42
        
        # Mock the peek method for sample filenames
        mock_collection.peek.return_value = {
            "metadatas": [
                {"filename": "test1.pdf"},
                {"filename": "test2.pdf"}
            ]
        }
        
        # Create retriever instance and call get_collection_stats
        retriever = ChromaRetriever()
        stats = retriever.get_collection_stats()
        
        # Assert stats results
        assert stats["document_count"] == 42
        assert "sample_filenames" in stats
        assert isinstance(stats["sample_filenames"], list)

class TestAzureSearchRetriever:
    """Tests for the AzureSearchRetriever class."""
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up environment variables for testing."""
        monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "https://test.search.windows.net")
        monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "test-index")
        monkeypatch.setenv("AZURE_SEARCH_API_KEY", "test_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    
    @patch("src.retriever.azure_search_retriever.SearchClient")
    @patch("src.retriever.azure_search_retriever.AzureKeyCredential")
    def test_retrieve_context(self, mock_credential_class, mock_search_client_class, mock_env):
        """Test Azure Search retrieval functionality."""
        # Mock environment variables
        with patch.dict('os.environ', {
            'AZURE_SEARCH_SERVICE_ENDPOINT': 'https://test-search.search.windows.net',
            'AZURE_SEARCH_INDEX_NAME': 'test-index',
            'AZURE_SEARCH_API_KEY': 'test-key',
            'OPENAI_API_KEY': 'test-openai-key',
            'OPENAI_EMBEDDING_MODEL_NAME': 'text-embedding-ada-002'
        }):
            # Create and configure mock credential
            mock_credential = Mock()
            mock_credential_class.return_value = mock_credential

            # Create and configure mock search client
            mock_search_client = Mock()
            mock_search_client_class.return_value = mock_search_client
            
            # Ensure the mock client is properly initialized
            mock_search_client_class.assert_not_called()  # Should not be called before retriever creation

            # Create mock search result
            class MockSearchResult:
                def __init__(self):
                    self.data = {
                        "content": "test content",
                        "filename": "test.txt",
                        "page_number": 1
                    }

                def __getitem__(self, key):
                    return self.data[key]

                def get(self, key, default=None):
                    return self.data.get(key, default)

            # Configure mock search response
            mock_search_client.search.return_value = [MockSearchResult()]

            # Create retriever - this should trigger client initialization
            retriever = AzureSearchRetriever()

            # Verify client was properly initialized
            mock_search_client_class.assert_called_once_with(
                endpoint='https://test-search.search.windows.net',
                index_name='test-index',
                credential=mock_credential
            )

            # Mock embeddings generation
            with patch.object(retriever, 'generate_embeddings', return_value=[0.1, 0.2, 0.3]):
                results = retriever.retrieve_context(
                    search_queries=["test query"],
                    filenames=["test.txt"]
                )

            assert len(results) == 1
            assert results[0]["text"] == "test content"
            assert results[0]["page"] == 1
            assert results[0]["filename"] == "test.txt"

            # Verify search was called with correct parameters
            mock_search_client.search.assert_called_once()
            call_args = mock_search_client.search.call_args
            assert call_args[1]["search_text"] == "test query"
            assert call_args[1]["filter"] == "filename eq 'test.txt'"
            assert call_args[1]["select"] == ["content", "filename", "page_number"] 