import unittest
from unittest.mock import patch, MagicMock, Mock
import os
import sys
import json
from typing import Dict, List, Any

# Import the LLMWrappers class from the LangChain implementation
from src.llm_calls.llm_wrappers import LLMWrappers, SearchQueries, ContextDecision, FinalAnswer, Citation, TokenUsage

class TestLLMWrappersLangChain(unittest.TestCase):
    """Unit tests for LLMWrappers with LangChain structured output."""
    
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
    
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_ensure_models_initialized(self, mock_chat_openai):
        """Test initialization of ChatOpenAI model."""
        # Setup mock
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        
        # Make sure the chat model is None before the test
        self.llm_wrapper.chat_model = None
        
        # Call method
        result = self.llm_wrapper._ensure_models_initialized()
        
        # Assert model was created correctly
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4o",
            temperature=0.0
        )
        self.assertEqual(self.llm_wrapper.chat_model, mock_chat_model)
        self.assertEqual(result, mock_chat_model)
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_token_usage_tracking(self, mock_chat_openai, mock_get_callback):
        """Test that token usage is properly tracked across multiple calls."""
        # Setup mocks
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = structured_llm
        
        # Setup callback mock
        mock_callback = MagicMock()
        mock_callback.prompt_tokens = 100
        mock_callback.completion_tokens = 50
        mock_callback.total_tokens = 150
        mock_callback.__enter__.return_value = mock_callback
        mock_get_callback.return_value = mock_callback
        
        # Setup return value for structured_llm.invoke
        mock_result = MagicMock()
        mock_result.queries = ["query 1", "query 2"]
        structured_llm.invoke.return_value = mock_result
        
        # Initial token usage should be zero
        self.assertEqual(self.llm_wrapper.get_token_usage(), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
        
        # Make first call
        self.llm_wrapper.generate_search_queries_llm(
            original_query="test query",
            retrieved_context=[],
            previous_queries=[],
            generation_attempt=1
        )
        
        # Check token usage after first call
        first_usage = self.llm_wrapper.get_token_usage()
        self.assertEqual(first_usage["prompt_tokens"], 100)
        self.assertEqual(first_usage["completion_tokens"], 50)
        self.assertEqual(first_usage["total_tokens"], 150)
        
        # Make second call
        self.llm_wrapper.generate_search_queries_llm(
            original_query="test query 2",
            retrieved_context=[],
            previous_queries=[],
            generation_attempt=1
        )
        
        # Check token usage is cumulative
        second_usage = self.llm_wrapper.get_token_usage()
        self.assertEqual(second_usage["prompt_tokens"], 200)
        self.assertEqual(second_usage["completion_tokens"], 100)
        self.assertEqual(second_usage["total_tokens"], 300)
        
        # Test reset
        self.llm_wrapper.reset_token_usage()
        reset_usage = self.llm_wrapper.get_token_usage()
        self.assertEqual(reset_usage["prompt_tokens"], 0)
        self.assertEqual(reset_usage["completion_tokens"], 0)
        self.assertEqual(reset_usage["total_tokens"], 0)
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_search_queries_llm(self, mock_chat_openai, mock_get_callback):
        """Test generating search queries with structured output."""
        # Setup mocks
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = structured_llm
        
        # Setup callback mock
        mock_callback = MagicMock()
        mock_callback.__enter__.return_value = mock_callback
        mock_get_callback.return_value = mock_callback
        
        # Setup return value for structured_llm.invoke
        mock_result = MagicMock()
        mock_result.queries = ["query 1", "query 2", "query 3"]
        structured_llm.invoke.return_value = mock_result
        
        # Call method
        result = self.llm_wrapper.generate_search_queries_llm(
            original_query="test query",
            retrieved_context=[],
            previous_queries=[],
            generation_attempt=1
        )
        
        # Assert correct return value
        self.assertEqual(result, ["query 1", "query 2", "query 3"])
        
        # Assert model was called correctly
        mock_chat_model.with_structured_output.assert_called_once_with(
            SearchQueries,
            method="function_calling"
        )
        structured_llm.invoke.assert_called_once()
        args, _ = structured_llm.invoke.call_args
        self.assertEqual(len(args[0]), 2)  # System and user messages
        self.assertEqual(args[0][0]["role"], "system")
        self.assertEqual(args[0][1]["role"], "user")
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_grade_context_llm(self, mock_chat_openai, mock_get_callback):
        """Test grading context with structured output."""
        # Setup mocks
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = structured_llm
        
        # Setup return value for structured_llm.invoke
        mock_result = MagicMock()
        mock_result.decision = "CONTINUE"
        structured_llm.invoke.return_value = mock_result
        
        # Call method
        result = self.llm_wrapper.grade_context_llm(
            original_query="test query",
            retrieved_context=[],
            iterations=1,
            max_iterations=5
        )
        
        # Assert correct return value
        self.assertEqual(result, "CONTINUE")
        
        # Assert model was called correctly
        self.llm_wrapper.chat_model.with_structured_output.assert_called_once_with(
            ContextDecision,
            method="function_calling"
        )
        structured_llm.invoke.assert_called_once()
        args, _ = structured_llm.invoke.call_args
        self.assertEqual(len(args[0]), 2)  # System and user messages
        self.assertEqual(args[0][0]["role"], "system")
        self.assertEqual(args[0][1]["role"], "user")
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_llm(self, mock_chat_openai, mock_get_callback):
        """Test generating final answer with structured output."""
        # Setup mocks
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = structured_llm
        
        # Setup return value for structured_llm.invoke
        mock_result = MagicMock()
        mock_result.answer = "This is the answer with citation [1]."
        mock_citation = MagicMock()
        mock_citation.text = "Source text"
        mock_citation.page = 5
        mock_citation.filename = "test_doc.pdf"
        mock_result.citations = [mock_citation]
        structured_llm.invoke.return_value = mock_result
        
        # Setup test context
        test_context = [
            {"text": "Context text", "page": 5, "filename": "test_doc.pdf"}
        ]
        
        # Call method
        answer, citations = self.llm_wrapper.generate_final_answer_llm(
            original_query="test query",
            retrieved_context=test_context
        )
        
        # Assert correct return values
        self.assertEqual(answer, "This is the answer with citation [1].")
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["text"], "Source text")
        self.assertEqual(citations[0]["page"], 5)
        self.assertEqual(citations[0]["filename"], "test_doc.pdf")
        
        # Assert model was called correctly
        self.llm_wrapper.chat_model.with_structured_output.assert_called_once_with(
            FinalAnswer,
            method="function_calling"
        )
        structured_llm.invoke.assert_called_once()
        args, _ = structured_llm.invoke.call_args
        self.assertEqual(len(args[0]), 2)  # System and user messages
        self.assertEqual(args[0][0]["role"], "system")
        self.assertEqual(args[0][1]["role"], "user")
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_empty_context(self, mock_chat_openai, mock_get_callback):
        """Test generating final answer with empty context."""
        # Call method with empty context
        answer, citations = self.llm_wrapper.generate_final_answer_llm(
            original_query="test query",
            retrieved_context=[]
        )
        
        # Assert information not found response
        self.assertEqual(answer, "Information not found in provided documents")
        self.assertEqual(citations, [])
    
    @patch('src.llm_calls.llm_wrappers.get_openai_callback')
    @patch('src.llm_calls.llm_wrappers.ChatOpenAI')
    def test_generate_final_answer_information_not_found(self, mock_chat_openai, mock_get_callback):
        """Test generating final answer when information is not found."""
        # Setup mocks
        mock_chat_model = MagicMock()
        mock_chat_openai.return_value = mock_chat_model
        structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = structured_llm
        
        # Setup return value indicating information not found
        mock_result = MagicMock()
        mock_result.answer = "Information not found in provided documents"
        mock_result.citations = [MagicMock()]  # This should be ignored
        structured_llm.invoke.return_value = mock_result
        
        # Setup test context
        test_context = [
            {"text": "Context text", "page": 5, "filename": "test_doc.pdf"}
        ]
        
        # Call method
        answer, citations = self.llm_wrapper.generate_final_answer_llm(
            original_query="test query",
            retrieved_context=test_context
        )
        
        # Assert correct return values - citations should be empty
        self.assertEqual(answer, "Information not found in provided documents")
        self.assertEqual(citations, [])

if __name__ == '__main__':
    unittest.main() 