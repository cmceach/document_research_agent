"""
Unit tests for graph node functions
"""
import pytest
from unittest.mock import patch, MagicMock
from src.graph_nodes.node_functions import (
    start_node_func,
    generate_search_queries_node_func,
    retrieve_context_node_func,
    grade_context_node_func,
    generate_final_answer_node_func,
    handle_failure_node_func
)
from src.llm_calls.llm_wrappers import LLMWrappers

def test_start_node_func():
    """Test that start_node_func initializes state correctly."""
    # Create test state with LLMWrappers
    llm_wrapper = LLMWrappers(lazy_init=True)
    initial_state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": llm_wrapper
    }
    
    # Call function
    result = start_node_func(initial_state)
    
    # Verify initial values
    assert result["iterations"] == 0
    assert result["max_iterations"] == 5
    assert result["generation_attempts"] == 0
    assert result["max_generation_attempts"] == 3
    assert result["search_queries"] == []
    assert result["retrieved_context"] == []
    assert result["agent_scratchpad"] == ""
    assert result["final_answer"] is None
    assert result["citations"] == []
    assert result["is_finished"] is False

@patch('src.graph_nodes.node_functions.retriever')
def test_generate_search_queries_node_func(mock_retriever):
    """Test that generate_search_queries_node_func uses llm_wrapper correctly."""
    # Create mock LLMWrappers
    mock_llm = MagicMock()
    mock_llm.generate_search_queries_llm.return_value = ["query 1", "query 2"]
    
    # Create test state
    state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": mock_llm,
        "iterations": 0,
        "generation_attempts": 0,
        "search_queries": [],
        "retrieved_context": [],
        "agent_scratchpad": ""
    }
    
    # Call function
    result = generate_search_queries_node_func(state)
    
    # Verify LLM was called correctly
    mock_llm.generate_search_queries_llm.assert_called_once_with(
        original_query="test query",
        retrieved_context=[],
        previous_queries=[],
        generation_attempt=1
    )
    
    # Verify result
    assert result["search_queries"] == ["query 1", "query 2"]
    assert result["generation_attempts"] == 1
    assert "Generated search queries" in result["agent_scratchpad"]

@patch('src.graph_nodes.node_functions.retriever')
def test_grade_context_node_func(mock_retriever):
    """Test that grade_context_node_func uses llm_wrapper correctly."""
    # Create mock LLMWrappers
    mock_llm = MagicMock()
    mock_llm.grade_context_llm.return_value = "CONTINUE"
    
    # Create test state
    state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": mock_llm,
        "iterations": 1,
        "max_iterations": 5,
        "generation_attempts": 1,
        "retrieved_context": [{"text": "test", "page": 1, "filename": "test.pdf"}],
        "agent_scratchpad": ""
    }
    
    # Call function
    result = grade_context_node_func(state)
    
    # Verify LLM was called correctly
    mock_llm.grade_context_llm.assert_called_once_with(
        original_query="test query",
        retrieved_context=[{"text": "test", "page": 1, "filename": "test.pdf"}],
        iterations=1,
        max_iterations=5
    )
    
    # Verify result
    assert result["context_decision"] == "CONTINUE"
    assert "Context evaluation decision" in result["agent_scratchpad"]

def test_generate_final_answer_node_func():
    """Test that generate_final_answer_node_func uses llm_wrapper correctly."""
    # Create mock LLMWrappers
    mock_llm = MagicMock()
    mock_llm.generate_final_answer_llm.return_value = (
        "The answer is 42",
        [{"text": "test citation", "page": 1, "filename": "test.pdf"}]
    )
    
    # Create test state
    state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": mock_llm,
        "retrieved_context": [{"text": "test", "page": 1, "filename": "test.pdf"}],
        "agent_scratchpad": ""
    }
    
    # Call function
    result = generate_final_answer_node_func(state)
    
    # Verify LLM was called correctly
    mock_llm.generate_final_answer_llm.assert_called_once_with(
        original_query="test query",
        retrieved_context=[{"text": "test", "page": 1, "filename": "test.pdf"}]
    )
    
    # Verify result
    assert result["final_answer"] == "The answer is 42"
    assert result["citations"] == [{"text": "test citation", "page": 1, "filename": "test.pdf"}]
    assert result["is_finished"] is True
    assert "Generating final answer" in result["agent_scratchpad"] 