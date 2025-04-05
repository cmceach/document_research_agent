"""
Unit tests for main module
"""
import pytest
import os
import sys
import argparse
from unittest.mock import patch, MagicMock
from src.main import check_collection_status, parse_arguments, main

# Test collection status check
@patch('src.main.ChromaRetriever')
def test_check_collection_status(mock_retriever_class):
    """Test the check_collection_status function."""
    # Setup mock
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    
    # Set up method returns for a direct path to success
    mock_retriever.get_collection_stats.return_value = {
        "collection_name": "test_collection",
        "document_count": 100,
        "embedding_model": "text-embedding-ada-002"
    }
    
    mock_retriever.retrieve_context.return_value = [
        {"text": "test document", "page": 1, "filename": "test.pdf"}
    ]
    
    # Call function with test files
    filenames = ["test.pdf", "other.pdf"]
    result = check_collection_status(filenames)
    
    # Assert result
    assert result is True, "check_collection_status should return True for a valid collection"

# Test collection status check with empty collection
@patch('src.main.ChromaRetriever')
def test_check_collection_status_empty(mock_retriever_class):
    """Test the check_collection_status function with an empty collection."""
    # Setup mock
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    
    # Set up method returns for the empty collection path
    mock_retriever.get_collection_stats.return_value = {
        "collection_name": "test_collection",
        "document_count": 0,
        "embedding_model": "text-embedding-ada-002"
    }
    
    # Mock empty retrieve_context
    mock_retriever.retrieve_context.return_value = []
    
    # Call function with test files
    filenames = ["test.pdf"]
    result = check_collection_status(filenames)
    
    # Assert result
    assert result is False, "check_collection_status should return False for an empty collection"

# Test argument parsing
def test_parse_arguments():
    """Test the argument parsing function."""
    # Test with valid arguments
    test_args = ["What is the meaning of life?", "--filenames", "file1.pdf", "file2.pdf"]
    with patch.object(sys, 'argv', ['src.main.py'] + test_args):
        args = parse_arguments()
        assert args.query == "What is the meaning of life?"
        assert args.filenames == ["file1.pdf", "file2.pdf"]
        assert args.verbose is False
        assert args.check_collection is False
    
    # Test with check-collection flag
    test_args = ["What is the meaning of life?", "--filenames", "file1.pdf", "--check-collection"]
    with patch.object(sys, 'argv', ['src.main.py'] + test_args):
        args = parse_arguments()
        assert args.query == "What is the meaning of life?"
        assert args.filenames == ["file1.pdf"]
        assert args.check_collection is True

# Test main function
@patch('src.main.check_collection_status')
@patch('src.main.build_graph')
@patch('src.main.parse_arguments')
def test_main_with_check_collection(mock_parse_args, mock_build_graph, mock_check_collection):
    """Test the main function with check_collection flag."""
    # Setup mock args
    mock_args = MagicMock()
    mock_args.check_collection = True
    mock_args.query = "test query"
    mock_args.filenames = ["file1.pdf"]
    mock_parse_args.return_value = mock_args
    
    # Mock check_collection to return True (success)
    mock_check_collection.return_value = True
    
    # Call main
    main()
    
    # Assert calls
    mock_parse_args.assert_called_once()
    mock_check_collection.assert_called_once_with(["file1.pdf"])
    # Verify build_graph is not called when check_collection is True
    mock_build_graph.assert_not_called()

@patch('src.main.check_collection_status')
@patch('src.main.build_graph')
@patch('src.main.parse_arguments')
@patch('src.llm_calls.llm_wrappers.LLMWrappers')
def test_main_normal_run(mock_llm_wrappers, mock_parse_args, mock_build_graph, mock_check_collection):
    """Test the main function for normal agent execution."""
    # Setup mock args
    mock_args = MagicMock()
    mock_args.check_collection = False
    mock_args.query = "test query"
    mock_args.filenames = ["file1.pdf"]
    mock_args.verbose = True
    mock_args.output = None
    mock_parse_args.return_value = mock_args
    
    # Mock LLMWrappers
    mock_llm = MagicMock()
    mock_llm_wrappers.return_value = mock_llm
    
    # Mock graph
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph
    mock_graph.invoke.return_value = {
        "final_answer": "42",
        "citations": []
    }
    
    # Call main with mocked print
    with patch('builtins.print'):
        main()
    
    # Assert calls
    mock_parse_args.assert_called_once()
    mock_check_collection.assert_not_called()
    mock_build_graph.assert_called_once()
    
    # Verify graph was called with correct state including llm_wrapper
    mock_graph.invoke.assert_called_once()
    call_args = mock_graph.invoke.call_args[0][0]
    assert "original_query" in call_args
    assert "filenames" in call_args
    assert "llm_wrapper" in call_args
    assert call_args["original_query"] == "test query"
    assert call_args["filenames"] == ["file1.pdf"]
    assert call_args["llm_wrapper"] == mock_llm 