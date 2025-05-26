#!/usr/bin/env python3
"""
DocumentResearchAgent class that provides a simple interface for using the Document Research Agent.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Set, Union

from src.graph_builder import build_graph, invoke_graph
from src.retriever.chroma_retriever import ChromaRetriever
from src.llm_calls.llm_wrappers import LLMWrappers
from src.utils.env_utils import check_required_env_vars

# Setup logging
logger = logging.getLogger(__name__)

class DocumentResearchAgent:
    """A document research agent that can retrieve information from documents based on a query."""
    
    # Required environment variables
    REQUIRED_ENV_VARS = [
        "OPENAI_API_KEY",
        "CHROMA_DB_PATH",
        "CHROMA_COLLECTION_NAME"
    ]
    
    def __init__(self, lazy_init: bool = False):
        """
        Initialize the Document Research Agent.
        
        Args:
            lazy_init: Whether to delay initialization of components
        """
        # Check environment variables first
        if not check_required_env_vars(self.REQUIRED_ENV_VARS):
            raise EnvironmentError("Missing required environment variables")
            
        # Initialize components
        self.retriever = None
        self.llm_wrappers = None
        
        if not lazy_init:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the retriever and LLM wrapper components."""
        try:
            from src.retriever.chroma_retriever import ChromaRetriever
            from src.llm_calls.llm_wrappers import LLMWrappers
            from src.graph_builder import build_graph
            
            self.retriever = ChromaRetriever()
            self.llm_wrappers = LLMWrappers()
            self.graph = build_graph()
            logger.info("Successfully initialized agent components")
            
        except Exception as e:
            logger.error(f"Error initializing agent components: {e}")
            raise
    
    def check_collection_status(self, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check if the Chroma DB collection exists and has documents.
        
        Args:
            filenames: List of document filenames to check for (optional)
            
        Returns:
            Dictionary with status information about the collection
        """
        try:
            # Get collection statistics
            stats = self.retriever.get_collection_stats()
            
            result = {
                "success": True,
                "document_count": stats["document_count"],
                "embedding_count": stats.get("embedding_count", 0),
                "has_documents": stats["document_count"] > 0,
                "filenames_found": None
            }
            
            if stats["document_count"] == 0:
                logger.error("Collection exists but contains no documents.")
                result["success"] = False
                result["error"] = "Collection exists but contains no documents"
                return result
            
            # If filenames are provided, check if they exist in the collection
            if filenames:
                # Test retrieval with a basic query to check for specified filenames
                test_query = "test query"
                results = self.retriever.retrieve_context(
                    search_queries=[test_query],
                    filenames=filenames,
                    top_k=1
                )
                
                if not results:
                    logger.warning(f"No documents found matching the specified filenames: {filenames}")
                    result["success"] = False
                    result["error"] = f"No documents found matching the specified filenames: {filenames}"
                    result["filenames_found"] = False
                else:
                    result["filenames_found"] = True
            
            logger.info(f"Collection check completed: {stats['document_count']} total documents")
            return result
                
        except Exception as e:
            logger.error(f"Error checking collection status: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_count": 0,
                "has_documents": False
            }
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get the current token usage statistics
        
        Returns:
            Dictionary with token usage information
        """
        return self.llm_wrappers.get_token_usage()
    
    def reset_token_usage(self) -> None:
        """Reset the token usage counters to zero"""
        self.llm_wrappers.reset_token_usage()
    
    def run(
        self, 
        query: str, 
        filenames: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        include_scratchpad: bool = False,
        reset_token_usage: bool = True
    ) -> Dict[str, Any]:
        """
        Run the Document Research Agent with the given query.
        
        Args:
            query: The question to research in the documents
            filenames: List of document filenames to search within (optional)
            max_iterations: Maximum number of iterations (optional, overrides default)
            include_scratchpad: Whether to include agent reasoning in the output
            reset_token_usage: Whether to reset token usage counters before running
            
        Returns:
            Dictionary containing the final answer, citations, and token usage
        """
        # Reset token usage if requested
        if reset_token_usage:
            self.reset_token_usage()
            
        if not query or not isinstance(query, str) or query.strip() == "":
            logger.error("Invalid query: Query must be a non-empty string")
            return {
                "success": False,
                "error": "Query must be a non-empty string",
                "final_answer": "Error: Invalid query",
                "citations": [],
                "token_usage": self.get_token_usage()
            }
        
        try:
            # Prepare the input state
            input_state = {
                "original_query": query,
                "filenames": filenames or [],
                "llm_wrapper": self.llm_wrappers  # Add LLMWrappers instance to state
            }
            
            # Add optional max_iterations if provided
            if max_iterations is not None:
                if isinstance(max_iterations, int) and max_iterations > 0:
                    input_state["max_iterations"] = max_iterations
                else:
                    logger.warning(f"Invalid max_iterations: {max_iterations}, using default")
            
            logger.info(f"Starting research with query: {query}")
            if filenames:
                logger.info(f"Document filenames: {filenames}")
            
            # Execute the graph using invoke_graph
            final_state = invoke_graph(self.graph, input_state)
            
            # Get token usage information
            token_usage = self.get_token_usage()
            logger.info(f"Token usage - Total: {token_usage['total_tokens']}, "
                       f"Prompt: {token_usage['prompt_tokens']}, "
                       f"Completion: {token_usage['completion_tokens']}")
            
            # Prepare the result dictionary
            result = {
                "success": True,
                "final_answer": final_state.get("final_answer", "No answer generated"),
                "citations": final_state.get("citations", []),
                "token_usage": token_usage,
                "iterations": final_state.get("iterations", 0),
                "runtime": final_state.get("runtime", {})
            }
            
            # Include agent scratchpad if requested
            if include_scratchpad:
                result["agent_scratchpad"] = final_state.get("agent_scratchpad", "")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running Document Research Agent: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "final_answer": "Error occurred during research",
                "citations": [],
                "token_usage": self.get_token_usage()
            } 