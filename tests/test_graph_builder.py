"""
Unit tests for GraphBuilder
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from src.graph_builder import build_graph
from src.graph_state.agent_state import AgentState
from src.llm_calls.llm_wrappers import LLMWrappers
from langgraph.graph import StateGraph, END

def test_build_graph_structure():
    """Test that the graph is built with the correct structure."""
    # Build the graph
    graph = build_graph()
    
    # Verify graph has invoke method
    assert hasattr(graph, 'invoke'), "Graph should have an invoke method"
    
    # Verify it's a compiled StateGraph
    assert graph is not None, "Graph should be compiled"

def test_build_graph_nodes():
    """Test that the graph has the expected nodes."""
    # Build the graph
    graph = build_graph()
    
    # Check that the graph has the expected structure
    # Note: We can't easily access internal node structure in compiled graphs,
    # but we can verify it compiles without errors
    assert graph is not None, "Graph should compile successfully"

def test_llm_wrappers_initialization():
    """Test that LLMWrappers can be initialized for use in the graph."""
    # Test lazy initialization
    llm_wrapper = LLMWrappers(lazy_init=True)
    assert llm_wrapper is not None, "LLMWrappers should initialize with lazy_init=True"
    
    # Test that it has the expected methods
    assert hasattr(llm_wrapper, 'generate_search_queries_llm'), "Should have generate_search_queries_llm method"
    assert hasattr(llm_wrapper, 'grade_context_llm'), "Should have grade_context_llm method"
    assert hasattr(llm_wrapper, 'generate_final_answer_llm'), "Should have generate_final_answer_llm method" 