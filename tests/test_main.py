"""
Unit tests for main module
"""
import pytest
import os
import sys
import argparse
from unittest.mock import patch, MagicMock
from src.main import parse_arguments, main, check_document_availability, print_result

# Test document availability check
def test_check_document_availability():
    """Test the check_document_availability function."""
    # Test with non-existent files
    filenames = ["nonexistent1.pdf", "nonexistent2.pdf"]
    result = check_document_availability(filenames)
    assert result is False, "check_document_availability should return False for non-existent files"

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

# Test print_result function
def test_print_result():
    """Test the print_result function."""
    result = {
        "success": True,
        "final_answer": "Test answer",
        "citations": [
            {"filename": "test.pdf", "page": 1, "text": "Test citation"}
        ],
        "iterations": 2
    }
    
    # Test that print_result doesn't raise an exception
    with patch('builtins.print'):
        print_result(result, verbose=False)
        print_result(result, verbose=True)

# Test main function with check collection
@patch('src.main.DocumentResearchAgent')
@patch('src.main.parse_arguments')
def test_main_with_check_collection(mock_parse_args, mock_agent_class):
    """Test the main function with check_collection flag."""
    # Setup mock args
    mock_args = MagicMock()
    mock_args.check_collection = True
    mock_args.query = "test query"
    mock_args.filenames = ["file1.pdf"]
    mock_args.debug_retrieval = False
    mock_parse_args.return_value = mock_args
    
    # Setup mock agent
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    mock_agent.check_collection_status.return_value = {
        "success": True,
        "document_count": 10
    }
    
    # Call main
    with patch('builtins.print'):
        result = main()
    
    # Assert calls
    mock_parse_args.assert_called_once()
    mock_agent.check_collection_status.assert_called_once_with(["file1.pdf"])
    assert result == 0

# Test main function normal run
@patch('src.main.DocumentResearchAgent')
@patch('src.main.parse_arguments')
def test_main_normal_run(mock_parse_args, mock_agent_class):
    """Test the main function for normal agent execution."""
    # Setup mock args
    mock_args = MagicMock()
    mock_args.check_collection = False
    mock_args.query = "test query"
    mock_args.filenames = ["file1.pdf"]
    mock_args.verbose = True
    mock_args.output = None
    mock_args.debug_retrieval = False
    mock_args.max_iterations = None
    mock_parse_args.return_value = mock_args
    
    # Setup mock agent
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    mock_agent.run.return_value = {
        "success": True,
        "final_answer": "42",
        "citations": []
    }
    
    # Call main with mocked print
    with patch('builtins.print'):
        with patch('src.main.check_document_availability', return_value=True):
            result = main()
    
    # Assert calls
    mock_parse_args.assert_called_once()
    mock_agent.run.assert_called_once_with(
        query="test query",
        filenames=["file1.pdf"],
        max_iterations=None,
        include_scratchpad=True
    )
    assert result == 0 