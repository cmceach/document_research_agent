"""
Unit tests for LLM wrappers
"""
import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from src.llm_calls.llm_wrappers import LLMWrappers

@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_generate_search_queries_llm(mock_openai):
    """Test the generate_search_queries_llm function."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Setup mock response
    mock_choice = MagicMock()
    mock_choice.message.content = '["query 1", "query 2", "query 3"]'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Initialize LLMWrappers
    llm = LLMWrappers()
    
    # Call function
    result = llm.generate_search_queries_llm(
        original_query="What is the meaning of life?",
        retrieved_context=[],
        previous_queries=[],
        generation_attempt=1
    )
    
    # Assert result
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "query 1"
    
    # Verify chat completion was called with appropriate parameters
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == llm.chat_model
    assert len(call_args["messages"]) == 2
    assert call_args["max_completion_tokens"] == 300

@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_generate_search_queries_llm_invalid_json(mock_openai):
    """Test handling of invalid JSON responses in generate_search_queries_llm."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Setup mock response with non-JSON content
    mock_choice = MagicMock()
    mock_choice.message.content = 'This is not valid JSON but contains ["query 1", "query 2"]'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Initialize LLMWrappers
    llm = LLMWrappers()
    
    # Call function
    result = llm.generate_search_queries_llm(
        original_query="What is the meaning of life?",
        retrieved_context=[],
        previous_queries=[],
        generation_attempt=1
    )
    
    # Assert result still contains at least one query
    assert isinstance(result, list)
    assert len(result) > 0
    
@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_grade_context_llm(mock_openai):
    """Test the grade_context_llm function."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Setup mock responses for different decisions
    decisions = ["FINISH", "CONTINUE", "RETRY_GENERATION", "FAIL"]
    
    for decision in decisions:
        # Create a new mock response for each decision
        mock_choice = MagicMock()
        mock_choice.message.content = decision
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Initialize LLMWrappers
        llm = LLMWrappers()
        
        # Call function
        result = llm.grade_context_llm(
            original_query="What is the meaning of life?",
            retrieved_context=[{"text": "test content", "page": 1, "filename": "test.pdf"}],
            iterations=1,
            max_iterations=5
        )
        
        # Assert result matches expected decision
        assert result == decision
        
@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_generate_final_answer_llm(mock_openai):
    """Test the generate_final_answer_llm function."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Setup mock response with valid JSON
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "answer": "The answer is 42 [1]",
        "citations": [
            {"text": "The meaning of life is 42", "page": 1, "filename": "hitchhiker.pdf"}
        ]
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Initialize LLMWrappers
    llm = LLMWrappers()
    
    # Call function
    answer, citations = llm.generate_final_answer_llm(
        original_query="What is the meaning of life?",
        retrieved_context=[{"text": "The meaning of life is 42", "page": 1, "filename": "hitchhiker.pdf"}]
    )
    
    # Assert results
    assert answer == "The answer is 42 [1]"
    assert len(citations) == 1
    assert citations[0]["text"] == "The meaning of life is 42"
    assert citations[0]["page"] == 1
    assert citations[0]["filename"] == "hitchhiker.pdf"
    
    # Verify chat completion was called with appropriate parameters
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == llm.chat_model
    assert len(call_args["messages"]) == 2
    assert call_args["max_completion_tokens"] == 1500

@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_generate_final_answer_llm_empty_context(mock_openai):
    """Test generate_final_answer_llm with empty context."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Initialize LLMWrappers
    llm = LLMWrappers()
    
    # Call function with empty context
    answer, citations = llm.generate_final_answer_llm(
        original_query="What is the meaning of life?",
        retrieved_context=[]
    )
    
    # Assert results
    assert answer == "Information not found in provided documents"
    assert len(citations) == 0
    
    # Verify chat completion was not called
    mock_client.chat.completions.create.assert_not_called()
    
@patch('src.llm_calls.llm_wrappers.OpenAI')
def test_error_handling_in_client_initialization(mock_openai):
    """Test error handling in OpenAI client initialization."""
    # Setup mock to raise exception
    mock_openai.side_effect = Exception("Test error")
    
    # Test lazy initialization
    llm = LLMWrappers(lazy_init=True)
    
    # Verify client is None
    assert llm.client is None
    
    # Test that error is raised when using the client
    with pytest.raises(Exception):
        llm._ensure_client_initialized() 