import unittest
from unittest.mock import patch, MagicMock, Mock
import os
import sys
import json
from typing import Dict, List, Any

# Import the LLMWrappers class from the LangChain implementation
from src.llm_calls.llm_wrappers_langchain import LLMWrappers, SearchQueries, ContextDecision, FinalAnswer, Citation

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
    
    def test_ensure_models_initialized(self):
        """Test initialization of ChatOpenAI model."""
        # Setup mock for the ChatOpenAI import
        mock_chat_model = MagicMock()
        mock_chat_openai = MagicMock(return_value=mock_chat_model)
        
        # Make sure the chat model is None before the test
        self.llm_wrapper.chat_model = None
        
        # Mock the import inside the method
        with patch.dict('sys.modules', {'langchain_openai': MagicMock(ChatOpenAI=mock_chat_openai)}):
            # Call method
            result = self.llm_wrapper._ensure_models_initialized()
            
            # Assert model was created correctly
            mock_chat_openai.assert_called_once()
            mock_chat_openai.assert_called_with(model_name="gpt-4o", temperature=0.0)
            self.assertEqual(self.llm_wrapper.chat_model, mock_chat_model)
            self.assertEqual(result, mock_chat_model)
    
    @patch('src.llm_calls.llm_wrappers_langchain.LLMWrappers._ensure_models_initialized')
    def test_generate_search_queries_llm(self, mock_ensure_init):
        """Test generating search queries with structured output."""
        # Setup mocks
        self.llm_wrapper.chat_model = MagicMock()
        structured_llm = MagicMock()
        self.llm_wrapper.chat_model.with_structured_output.return_value = structured_llm
        
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
        self.llm_wrapper.chat_model.with_structured_output.assert_called_once_with(
            SearchQueries,
            method="function_calling"
        )
        structured_llm.invoke.assert_called_once()
        args, _ = structured_llm.invoke.call_args
        self.assertEqual(len(args[0]), 2)  # System and user messages
        self.assertEqual(args[0][0]["role"], "system")
        self.assertEqual(args[0][1]["role"], "user")
    
    @patch('src.llm_calls.llm_wrappers_langchain.LLMWrappers._ensure_models_initialized')
    def test_grade_context_llm(self, mock_ensure_init):
        """Test grading context with structured output."""
        # Setup mocks
        self.llm_wrapper.chat_model = MagicMock()
        structured_llm = MagicMock()
        self.llm_wrapper.chat_model.with_structured_output.return_value = structured_llm
        
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
    
    @patch('src.llm_calls.llm_wrappers_langchain.LLMWrappers._ensure_models_initialized')
    def test_generate_final_answer_llm(self, mock_ensure_init):
        """Test generating final answer with structured output."""
        # Setup mocks
        self.llm_wrapper.chat_model = MagicMock()
        structured_llm = MagicMock()
        self.llm_wrapper.chat_model.with_structured_output.return_value = structured_llm
        
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
    
    @patch('src.llm_calls.llm_wrappers_langchain.LLMWrappers._ensure_models_initialized')
    def test_generate_final_answer_empty_context(self, mock_ensure_init):
        """Test generating final answer with empty context."""
        # Call method with empty context
        answer, citations = self.llm_wrapper.generate_final_answer_llm(
            original_query="test query",
            retrieved_context=[]
        )
        
        # Assert information not found response
        self.assertEqual(answer, "Information not found in provided documents")
        self.assertEqual(citations, [])
    
    @patch('src.llm_calls.llm_wrappers_langchain.LLMWrappers._ensure_models_initialized')
    def test_generate_final_answer_information_not_found(self, mock_ensure_init):
        """Test generating final answer when information is not found."""
        # Setup mocks
        self.llm_wrapper.chat_model = MagicMock()
        structured_llm = MagicMock()
        self.llm_wrapper.chat_model.with_structured_output.return_value = structured_llm
        
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