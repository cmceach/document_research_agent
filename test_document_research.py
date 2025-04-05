#!/usr/bin/env python3
"""
Test script for the Document Research Agent.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from src.agent import DocumentResearchAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Document Research Agent")
    
    parser.add_argument(
        "--query",
        type=str,
        default="What are the key provisions of the Employment Non-Discrimination Act?",
        help="The question to research"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        action="append",
        help="Document filename(s) to search within (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--check-collection",
        action="store_true",
        help="Only check the collection status without running a query"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include agent scratchpad in output"
    )
    
    return parser.parse_args()

def test_document_research():
    """Test the Document Research Agent with command line arguments."""
    args = parse_args()
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1
    
    try:
        # Initialize agent
        agent = DocumentResearchAgent()
        
        # Check collection if requested
        if args.check_collection:
            logger.info("Checking ChromaDB collection status...")
            status = agent.check_collection_status(args.filename)
            if status["success"]:
                logger.info(f"Collection check successful: {status['document_count']} documents found")
                if args.filename and status["filenames_found"]:
                    logger.info(f"Specified filenames found in collection")
            else:
                logger.error(f"Collection check failed: {status.get('error', 'Unknown error')}")
                return 1
            return 0
        
        # Run the query
        query = args.query
        logger.info(f"Running agent with query: {query}")
        
        result = agent.run(
            query=query,
            filenames=args.filename,
            max_iterations=args.max_iterations,
            include_scratchpad=args.verbose
        )
        
        # Handle failure
        if not result["success"]:
            logger.error(f"Agent run failed: {result.get('error', 'Unknown error')}")
            return 1
        
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
        
        # Display scratchpad if requested
        if args.verbose and "agent_scratchpad" in result:
            logger.info("\n----- Agent Reasoning -----")
            logger.info(result["agent_scratchpad"])
            logger.info("-------------------------")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during test: {e}")
        return 1

if __name__ == "__main__":
    exit(test_document_research()) 