#!/usr/bin/env python3
"""
Comprehensive test script for the Document Research Agent.
Includes individual testing and sample query execution.
"""

import os
import sys
import logging
import argparse
import subprocess
from typing import List, Dict, Any
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

# Sample queries for comprehensive testing
SAMPLE_QUERIES = [
    {
        "query": "What are the confidentiality provisions in non-disclosure agreements?",
        "filenames": [
            "test_data/legal_document_03_non_disclosure_agreement.pdf",
            "test_data/legal_document_08_non_disclosure_agreement.pdf",
            "test_data/sample_nda.pdf"
        ]
    },
    {
        "query": "What are the termination clauses in employment contracts?",
        "filenames": [
            "test_data/legal_document_01_employment_contract.pdf",
            "test_data/legal_document_06_employment_contract.pdf"
        ]
    },
    {
        "query": "What are the payment terms in service agreements?",
        "filenames": [
            "test_data/legal_document_02_service_agreement.pdf",
            "test_data/legal_document_07_service_agreement.pdf"
        ]
    },
    {
        "query": "What are the default clauses in lease agreements?",
        "filenames": [
            "test_data/legal_document_04_lease_agreement.pdf",
            "test_data/legal_document_09_lease_agreement.pdf"
        ]
    },
    {
        "query": "What representations and warranties are included in purchase agreements?",
        "filenames": [
            "test_data/legal_document_05_purchase_agreement.pdf",
            "test_data/legal_document_10_purchase_agreement.pdf"
        ]
    },
    {
        "query": "What are the requirements for shuttle service in the contract?",
        "filenames": [
            "test_data/sample_contract_shuttle.pdf"
        ]
    }
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Document Research Agent")
    
    # Individual query options
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
    
    # Sample query options
    parser.add_argument(
        "--run-samples",
        action="store_true",
        help="Run all sample queries instead of individual query"
    )
    
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Run only a specific sample query by index (0-based)"
    )
    
    return parser.parse_args()

def run_sample_query(query: str, filenames: List[str], verbose: bool = False, check_collection: bool = False) -> None:
    """Run a sample query against the Document Research Agent."""
    # Build the command
    cmd = ["python", "-m", "src.main", query, "--filenames"]
    cmd.extend(filenames)
    
    if verbose:
        cmd.append("--verbose")
    
    if check_collection:
        cmd.append("--check-collection")
    
    # Print the command
    print("\n" + "=" * 80)
    if check_collection:
        print(f"CHECKING COLLECTION FOR: {query}")
    else:
        print(f"RUNNING QUERY: {query}")
    print(f"FILES: {', '.join([os.path.basename(f) for f in filenames])}")
    print("=" * 80)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running query: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def run_sample_queries(args):
    """Run sample queries against the Document Research Agent."""
    # Check if we should run a specific query
    if args.sample_index is not None:
        if 0 <= args.sample_index < len(SAMPLE_QUERIES):
            query_info = SAMPLE_QUERIES[args.sample_index]
            run_sample_query(
                query_info["query"], 
                query_info["filenames"], 
                args.verbose,
                args.check_collection
            )
        else:
            print(f"Invalid query index. Must be between 0 and {len(SAMPLE_QUERIES)-1}")
            return 1
    else:
        # Run all queries
        for i, query_info in enumerate(SAMPLE_QUERIES):
            print(f"\nRunning sample query {i}...")
            run_sample_query(
                query_info["query"], 
                query_info["filenames"], 
                args.verbose,
                args.check_collection
            )
    
    return 0

def run_individual_query(args):
    """Run the Document Research Agent with individual query."""
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

def main():
    """Main function to handle both individual and sample query testing."""
    args = parse_args()
    
    # Determine which mode to run
    if args.run_samples or args.sample_index is not None:
        return run_sample_queries(args)
    else:
        return run_individual_query(args)

if __name__ == "__main__":
    exit(main()) 