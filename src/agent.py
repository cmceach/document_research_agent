#!/usr/bin/env python3
"""
DocumentResearchAgent class that provides a simple interface for using the Document Research Agent.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from src.graph_builder import build_graph
from src.retriever.chroma_retriever import ChromaRetriever

# Setup logging
logger = logging.getLogger(__name__)

class DocumentResearchAgent:
    """A document research agent that can retrieve information from documents based on a query."""
    
    def __init__(self, check_environment: bool = True):
        """
        Initialize the Document Research Agent.
        
        Args:
            check_environment: Whether to check environment variables on initialization
        """
        # Build the graph
        self.graph = build_graph()
        
        # Check environment if requested
        if check_environment:
            self._check_environment()
        
        # Initialize retriever
        self.retriever = ChromaRetriever()
    
    def _check_environment(self) -> bool:
        """Check if all required environment variables are set."""
        required_vars = [
            "OPENAI_API_KEY",
            "CHROMA_DB_PATH",
            "CHROMA_COLLECTION_NAME",
            "OPENAI_EMBEDDING_MODEL_NAME"
        ]
        
        missing = [var for var in required_vars if not os.environ.get(var)]
        
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            logger.error("Please set these variables in your .env file or environment.")
            return False
        
        return True
    
    def check_collection_status(self, filenames: Optional[List[str]] = None) -> bool:
        """
        Check if the Chroma DB collection exists and has documents.
        
        Args:
            filenames: List of document filenames to check for (optional)
            
        Returns:
            Boolean indicating whether the collection check was successful
        """
        try:
            # Get collection statistics
            stats = self.retriever.get_collection_stats()
            
            if stats["document_count"] == 0:
                logger.error("Collection exists but contains no documents.")
                return False
            
            # If filenames are provided, check if they exist in the collection
            if filenames:
                # Test retrieval with a basic query to check for any documents with the specified filenames
                test_query = "test query"  # Not important, just checking if any documents match filenames
                results = self.retriever.retrieve_context(
                    search_queries=[test_query],
                    filenames=filenames,
                    top_k=1
                )
                
                if not results:
                    logger.warning(f"No documents found matching the specified filenames: {filenames}")
                    return False
            
            logger.info(f"Collection check successful: {stats['document_count']} total documents")
            return True
                
        except Exception as e:
            logger.error(f"Error checking collection status: {e}")
            return False
    
    def run(self, query: str, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the Document Research Agent with the given query.
        
        Args:
            query: The question to research in the documents
            filenames: List of document filenames to search within (optional)
            
        Returns:
            Dictionary containing the final answer and citations
        """
        try:
            # Prepare the input state
            input_state = {
                "original_query": query,
                "filenames": filenames or []
            }
            
            logger.info(f"Starting research with query: {query}")
            if filenames:
                logger.info(f"Document filenames: {filenames}")
            
            # Execute the graph
            final_state = self.graph.invoke(input_state)
            
            # Format the output
            result = {
                "final_answer": final_state.get("final_answer", "No answer generated"),
                "citations": final_state.get("citations", []),
                "agent_scratchpad": final_state.get("agent_scratchpad", "")
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing Document Research Agent: {e}")
            return {
                "final_answer": f"Error: {str(e)}",
                "citations": [],
                "agent_scratchpad": ""
            } 