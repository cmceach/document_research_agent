import pytest
from unittest.mock import Mock, patch
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