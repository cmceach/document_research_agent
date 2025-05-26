"""
Unit tests for LLM wrappers - consolidated tests for both basic and LangChain functionality
"""
import pytest
import unittest
import json
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any
from src.llm_calls.llm_wrappers import LLMWrappers, SearchQueries, ContextDecision, FinalAnswer, Citation, TokenUsage

class TestLLMWrappers(unittest.TestCase):
    """Unit tests for LLMWrappers with both basic and LangChain structured output."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_api_key',
            'OPENAI_CHAT_MODEL_NAME': 'gpt-4o'
        })
        self.env_patcher.start()
        
        # Create instance with lazy_init to avoid actual API calls
        self.llm_wrapper = LLMWrappers(lazy_init=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()

# LangChain-based tests (Current implementation)

class TestLLMWrappersLangChain(TestLLMWrappers):
    """Test LLMWrappers with LangChain structured output functionality."""
    
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_ensure_models_initialized(self, mock_chat_openai):
        """Test that models are initialized correctly."""
        # Setup mock
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        # Initialize LLMWrappers without lazy_init
        llm = LLMWrappers(lazy_init=False)
        
        # Verify ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(
            model_name='gpt-4o',
            temperature=0.0
        )
        
        # Verify model is set
        assert llm.chat_model == mock_model
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_token_usage_tracking(self, mock_chat_openai, mock_get_callback):
        """Test that token usage is tracked correctly."""
        # Setup mocks
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        # Mock the callback context manager
        mock_callback = MagicMock()
        mock_callback.__enter__ = MagicMock(return_value=mock_callback)
        mock_callback.__exit__ = MagicMock(return_value=None)
        mock_callback.total_tokens = 100
        mock_callback.prompt_tokens = 60
        mock_callback.completion_tokens = 40
        mock_get_callback.return_value = mock_callback
        
        # Mock the structured output
        mock_result = SearchQueries(queries=["test query"])
        mock_structured_model.invoke.return_value = mock_result
        
        # Initialize LLMWrappers
        llm = LLMWrappers(lazy_init=False)
        
        # Call a method that should track tokens
        result = llm.generate_search_queries_llm(
            original_query="test query",
            retrieved_context=[],
            previous_queries=[],
            generation_attempt=1
        )
        
        # Verify result
        assert result == ["test query"]
        
        # Verify token usage was tracked
        usage = llm.get_token_usage()
        assert usage["total_tokens"] == 100
        assert usage["prompt_tokens"] == 60
        assert usage["completion_tokens"] == 40
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_search_queries_llm_structured(self, mock_chat_openai, mock_get_callback):
        """Test generate_search_queries_llm with structured output."""
        # Setup mocks
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        # Mock the callback
        mock_callback = MagicMock()
        mock_callback.__enter__ = MagicMock(return_value=mock_callback)
        mock_callback.__exit__ = MagicMock(return_value=None)
        mock_callback.total_tokens = 50
        mock_callback.prompt_tokens = 30
        mock_callback.completion_tokens = 20
        mock_get_callback.return_value = mock_callback
        
        # Mock the structured output
        mock_result = SearchQueries(queries=["query 1", "query 2", "query 3"])
        mock_structured_model.invoke.return_value = mock_result
        
        # Initialize LLMWrappers
        llm = LLMWrappers(lazy_init=False)
        
        # Call function
        result = llm.generate_search_queries_llm(
            original_query="What is the meaning of life?",
            retrieved_context=[],
            previous_queries=[],
            generation_attempt=1
        )
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["query 1", "query 2", "query 3"]
        
        # Verify structured model was called
        mock_structured_model.invoke.assert_called_once()
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_grade_context_llm_structured(self, mock_chat_openai, mock_get_callback):
        """Test grade_context_llm with structured output."""
        # Setup mocks
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        # Mock the callback
        mock_callback = MagicMock()
        mock_callback.__enter__ = MagicMock(return_value=mock_callback)
        mock_callback.__exit__ = MagicMock(return_value=None)
        mock_callback.total_tokens = 30
        mock_callback.prompt_tokens = 20
        mock_callback.completion_tokens = 10
        mock_get_callback.return_value = mock_callback
        
        # Test different decisions
        decisions = ["FINISH", "CONTINUE", "RETRY_GENERATION", "FAIL"]
        
        for decision in decisions:
            # Mock the structured output
            mock_result = ContextDecision(decision=decision)
            mock_structured_model.invoke.return_value = mock_result
            
            # Initialize LLMWrappers
            llm = LLMWrappers(lazy_init=False)
            
            # Call function
            result = llm.grade_context_llm(
                original_query="What is the meaning of life?",
                retrieved_context=[{"text": "test content", "page": 1, "filename": "test.pdf"}],
                iterations=1,
                max_iterations=5
            )
            
            # Verify result
            assert result == decision
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_llm_structured(self, mock_chat_openai, mock_get_callback):
        """Test generate_final_answer_llm with structured output."""
        # Setup mocks
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        # Mock the callback
        mock_callback = MagicMock()
        mock_callback.__enter__ = MagicMock(return_value=mock_callback)
        mock_callback.__exit__ = MagicMock(return_value=None)
        mock_callback.total_tokens = 100
        mock_callback.prompt_tokens = 60
        mock_callback.completion_tokens = 40
        mock_get_callback.return_value = mock_callback
        
        # Mock the structured output
        citation = Citation(
            text="The meaning of life is 42",
            page=1,
            filename="hitchhiker.pdf"
        )
        mock_result = FinalAnswer(
            answer="The answer is 42 [1]",
            citations=[citation]
        )
        mock_structured_model.invoke.return_value = mock_result
        
        # Initialize LLMWrappers
        llm = LLMWrappers(lazy_init=False)
        
        # Call function
        answer, citations = llm.generate_final_answer_llm(
            original_query="What is the meaning of life?",
            retrieved_context=[{"text": "The meaning of life is 42", "page": 1, "filename": "hitchhiker.pdf"}]
        )
        
        # Verify results
        assert answer == "The answer is 42 [1]"
        assert len(citations) == 1
        assert citations[0]["text"] == "The meaning of life is 42"
        assert citations[0]["page"] == 1
        assert citations[0]["filename"] == "hitchhiker.pdf"
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_empty_context_structured(self, mock_chat_openai, mock_get_callback):
        """Test generate_final_answer_llm with empty context."""
        # Setup mocks (though they shouldn't be called)
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_get_callback.return_value = MagicMock()
        
        # Initialize LLMWrappers
        llm = LLMWrappers(lazy_init=False)
        
        # Call function with empty context
        answer, citations = llm.generate_final_answer_llm(
            original_query="What is the meaning of life?",
            retrieved_context=[]
        )
        
        # Verify results
        assert answer == "Information not found in provided documents"
        assert len(citations) == 0
        
        # Verify structured model was not called
        mock_model.with_structured_output.assert_not_called()
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_information_not_found(self, mock_chat_openai, mock_get_callback):
        """Test generate_final_answer_llm when LLM returns 'Information not found'."""
        # Setup mocks
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_chat_openai.return_value = mock_model
        
        # Mock the callback
        mock_callback = MagicMock()
        mock_callback.__enter__ = MagicMock(return_value=mock_callback)
        mock_callback.__exit__ = MagicMock(return_value=None)
        mock_callback.total_tokens = 50
        mock_callback.prompt_tokens = 30
        mock_callback.completion_tokens = 20
        mock_get_callback.return_value = mock_callback
        
        # Mock the structured output with "Information not found"
        mock_result = FinalAnswer(
            answer="Information not found in provided documents",
            citations=[]
        )
        mock_structured_model.invoke.return_value = mock_result
        
        # Initialize LLMWrappers
        llm = LLMWrappers(lazy_init=False)
        
        # Call function
        answer, citations = llm.generate_final_answer_llm(
            original_query="What is the meaning of life?",
            retrieved_context=[{"text": "Irrelevant content", "page": 1, "filename": "test.pdf"}]
        )
        
        # Verify results
        assert answer == "Information not found in provided documents"
        assert len(citations) == 0

if __name__ == '__main__':
    unittest.main() 