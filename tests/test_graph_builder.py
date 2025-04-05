"""
Unit tests for GraphBuilder
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from src.graph_builder import build_graph
from src.graph_state.agent_state import AgentState
from src.llm_calls.llm_wrappers import LLMWrappers
from langgraph.graph import StateGraph, END

@patch('src.graph_builder.handle_failure_node_func')
@patch('src.graph_builder.generate_final_answer_node_func')
@patch('src.graph_builder.grade_context_node_func')
@patch('src.graph_builder.retrieve_context_node_func')
@patch('src.graph_builder.generate_search_queries_node_func')
@patch('src.graph_builder.start_node_func')
def test_build_graph(mock_start, mock_generate_search_queries,
                    mock_retrieve_context, mock_grade_context,
                    mock_generate_final_answer, mock_handle_failure):
    """Test that the graph is built with the correct structure."""
    
    # Setup mock returns
    mock_start.return_value = {
        "iterations": 0,
        "max_iterations": 5,
        "generation_attempts": 0,
        "max_generation_attempts": 3,
        "search_queries": [],
        "retrieved_context": [],
        "agent_scratchpad": "",
        "final_answer": None,
        "citations": [],
        "is_finished": False
    }
    
    # First and second search query generation
    mock_generate_search_queries.side_effect = [
        {
            "search_queries": ["query 1"],
            "generation_attempts": 1,  # Increment from 0
            "agent_scratchpad": "Generated queries:\n- query 1"
        },
        {
            "search_queries": ["query 2"],
            "generation_attempts": 1,  # Increment from 0 after reset
            "agent_scratchpad": "Generated queries:\n- query 2"
        },
        {
            "search_queries": ["query 3"],
            "generation_attempts": 2,  # Increment from 1
            "agent_scratchpad": "Generated queries:\n- query 3"
        },
        {
            "search_queries": ["query 4"],
            "generation_attempts": 3,  # Max attempts reached
            "agent_scratchpad": "Generated queries:\n- query 4"
        }
    ]
    
    # First and second context retrieval - each call adds to retrieved_context
    mock_retrieve_context.side_effect = [
        {
            "iterations": 1,  # Increment from 0
            "agent_scratchpad": "Retrieved 1 new context items in iteration 1:\n[1] Source: test.pdf, Page: 1\nText: test...\n",
            "retrieved_context": [{"text": "test", "page": 1, "filename": "test.pdf"}]  # Will be appended
        },
        {
            "iterations": 2,  # Increment from 1
            "agent_scratchpad": "Retrieved 1 new context items in iteration 2:\n[1] Source: test.pdf, Page: 2\nText: more test...\n",
            "retrieved_context": [{"text": "more test", "page": 2, "filename": "test.pdf"}]  # Will be appended
        },
        {
            "iterations": 3,  # Increment from 2
            "agent_scratchpad": "Retrieved 1 new context items in iteration 3:\n[1] Source: test.pdf, Page: 3\nText: final test...\n",
            "retrieved_context": [{"text": "final test", "page": 3, "filename": "test.pdf"}]  # Will be appended
        },
        {
            "iterations": 4,  # Increment from 3
            "agent_scratchpad": "Retrieved 1 new context items in iteration 4:\n[1] Source: test.pdf, Page: 4\nText: last test...\n",
            "retrieved_context": [{"text": "last test", "page": 4, "filename": "test.pdf"}]  # Will be appended
        }
    ]
    
    # First and second context grading
    mock_grade_context.side_effect = [
        {
            "context_decision": "CONTINUE",  # First try to get more context
            "agent_scratchpad": "Context evaluation decision: CONTINUE",
            "generation_attempts": 1  # Keep current value
        },
        {
            "context_decision": "RETRY_GENERATION",  # Try again with new queries
            "agent_scratchpad": "Context evaluation decision: RETRY_GENERATION\nResetting search strategy for next iteration.\n",
            "generation_attempts": 0  # Reset to 0
        },
        {
            "context_decision": "CONTINUE",  # Get more context
            "agent_scratchpad": "Context evaluation decision: CONTINUE",
            "generation_attempts": 2  # Keep current value
        },
        {
            "context_decision": "FINISH",  # Then finish after getting more context
            "agent_scratchpad": "Context evaluation decision: FINISH",
            "generation_attempts": 3  # Keep current value
        }
    ]
    
    mock_generate_final_answer.return_value = {
        "final_answer": "The answer is 42",
        "agent_scratchpad": "Generating final answer...\n",
        "is_finished": True,
        "citations": [{"text": "test", "page": 1, "filename": "test.pdf"}]  # Will be appended to existing citations
    }
    
    # Build the graph
    graph = build_graph()
    
    # Verify graph has invoke method
    assert hasattr(graph, 'invoke'), "Graph should have an invoke method"
    
    # Create test input state with LLMWrappers
    llm_wrapper = LLMWrappers(lazy_init=True)
    input_state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": llm_wrapper,
        "iterations": 0,  # Initialize required fields
        "max_iterations": 5,
        "generation_attempts": 0,
        "max_generation_attempts": 3,
        "search_queries": [],
        "retrieved_context": [],  # Initialize empty list for operator.add field
        "agent_scratchpad": "",
        "final_answer": None,
        "citations": [],  # Initialize empty list for operator.add field
        "is_finished": False,
        "context_decision": "CONTINUE"  # Add initial decision
    }
    
    # Execute graph with debug mode and higher recursion limit
    result = graph.invoke(input_state, {"recursion_limit": 20, "debug": True})  # Increase recursion limit
    
    # Verify the expected sequence of calls
    mock_start.assert_called_once()
    assert mock_generate_search_queries.call_count == 4  # Should be called four times
    assert mock_retrieve_context.call_count == 4  # Should be called four times
    assert mock_grade_context.call_count == 4  # Should be called four times
    mock_generate_final_answer.assert_called_once()
    mock_handle_failure.assert_not_called()
    
    # Verify the final state
    assert result["is_finished"] is True, "Graph should finish execution"
    assert result["final_answer"] == "The answer is 42", "Final answer should be set"
    assert len(result["citations"]) > 0, "Citations should be populated"
    assert len(result["retrieved_context"]) == 4, "Should have retrieved context from all iterations"
    assert result["iterations"] == 4, "Should have completed 4 iterations"

@patch('src.graph_builder.handle_failure_node_func')
@patch('src.graph_builder.generate_final_answer_node_func')
@patch('src.graph_builder.grade_context_node_func')
@patch('src.graph_builder.retrieve_context_node_func')
@patch('src.graph_builder.generate_search_queries_node_func')
@patch('src.graph_builder.start_node_func')
def test_graph_failure_path(mock_start, mock_generate_search_queries,
                          mock_retrieve_context, mock_grade_context,
                          mock_generate_final_answer, mock_handle_failure):
    """Test that the graph handles failure paths correctly."""
    
    # Setup mock returns
    mock_start.return_value = {
        "iterations": 0,
        "max_iterations": 5,
        "generation_attempts": 0,
        "max_generation_attempts": 3,
        "search_queries": [],
        "retrieved_context": [],
        "agent_scratchpad": "",
        "final_answer": None,
        "citations": [],
        "is_finished": False
    }
    
    mock_generate_search_queries.return_value = {
        "search_queries": ["query 1"],
        "generation_attempts": 3,  # Max attempts reached
        "agent_scratchpad": "Generated queries",
        "iterations": 0,  # Preserve from start
        "max_iterations": 5,  # Preserve from start
        "max_generation_attempts": 3,  # Preserve from start
        "retrieved_context": [],  # Preserve from start
        "final_answer": None,  # Preserve from start
        "citations": [],  # Preserve from start
        "is_finished": False  # Preserve from start
    }
    
    mock_retrieve_context.return_value = {
        "retrieved_context": [],  # No context found
        "iterations": 1,
        "agent_scratchpad": "No context found",
        "search_queries": ["query 1"],  # Preserve from generate_search_queries
        "generation_attempts": 3,  # Preserve from generate_search_queries
        "max_iterations": 5,  # Preserve from start
        "max_generation_attempts": 3,  # Preserve from start
        "final_answer": None,  # Preserve from start
        "citations": [],  # Preserve from start
        "is_finished": False  # Preserve from start
    }
    
    mock_grade_context.return_value = {
        "context_decision": "FAIL",  # Trigger failure path
        "agent_scratchpad": "Failed to find relevant context",
        "generation_attempts": 3,  # Preserve from generate_search_queries
        "iterations": 1,  # Preserve from retrieve_context
        "max_iterations": 5,  # Preserve from start
        "max_generation_attempts": 3,  # Preserve from start
        "search_queries": ["query 1"],  # Preserve from generate_search_queries
        "retrieved_context": [],  # Preserve from retrieve_context
        "final_answer": None,  # Preserve from start
        "citations": [],  # Preserve from start
        "is_finished": False  # Preserve from start
    }
    
    mock_handle_failure.return_value = {
        "final_answer": "Information not found in provided documents",
        "citations": [],
        "is_finished": True,
        "agent_scratchpad": "Handling failure",
        "generation_attempts": 3,  # Preserve from generate_search_queries
        "iterations": 1,  # Preserve from retrieve_context
        "max_iterations": 5,  # Preserve from start
        "max_generation_attempts": 3,  # Preserve from start
        "search_queries": ["query 1"],  # Preserve from generate_search_queries
        "retrieved_context": []  # Preserve from retrieve_context
    }
    
    # Build the graph
    graph = build_graph()
    
    # Create test input state with LLMWrappers
    llm_wrapper = LLMWrappers(lazy_init=True)
    input_state = {
        "original_query": "test query",
        "filenames": ["test.pdf"],
        "llm_wrapper": llm_wrapper,
        "iterations": 0,  # Initialize required fields
        "max_iterations": 5,
        "generation_attempts": 0,
        "max_generation_attempts": 3,
        "search_queries": [],
        "retrieved_context": [],
        "agent_scratchpad": "",
        "final_answer": None,
        "citations": [],
        "is_finished": False
    }
    
    # Execute graph
    result = graph.invoke(input_state)
    
    # Verify the expected sequence of calls
    mock_start.assert_called_once()
    mock_generate_search_queries.assert_called_once()
    mock_retrieve_context.assert_called_once()
    mock_grade_context.assert_called_once()
    mock_generate_final_answer.assert_not_called()
    mock_handle_failure.assert_called_once()
    
    # Verify final state
    assert result["final_answer"] == "Information not found in provided documents"
    assert len(result["citations"]) == 0
    assert result["is_finished"] is True 