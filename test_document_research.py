#!/usr/bin/env python3
"""
Test script for the Document Research Agent with LangChain implementation.
"""

import os
import logging
from dotenv import load_dotenv
from src.agent import DocumentResearchAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_research():
    """Test the Document Research Agent with a sample query"""
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize agent
    agent = DocumentResearchAgent()
    
    # Test query
    query = "What are the key provisions of the Employment Non-Discrimination Act?"
    
    # Run agent
    logger.info(f"Running agent with query: {query}")
    result = agent.run(query)
    
    # Log result
    logger.info("Agent run completed successfully")
    logger.info(f"Final answer: {result['final_answer']}")
    
    # Display citations
    if result['citations']:
        logger.info("Citations:")
        for citation in result['citations']:
            logger.info(f"- {citation['filename']}, Page {citation['page']}: '{citation['text']}'")
    else:
        logger.info("No citations provided")

if __name__ == "__main__":
    test_document_research() 