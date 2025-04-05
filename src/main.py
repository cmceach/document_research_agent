import os
import json
import argparse
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.agent import DocumentResearchAgent

# Load environment variables
load_dotenv()

# Setup logging
log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document Research Agent')
    
    parser.add_argument(
        'query',
        type=str,
        help='The question to research in the documents'
    )
    
    parser.add_argument(
        '--filenames',
        type=str,
        nargs='+',
        required=True,
        help='List of document filenames to search within'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for the JSON result (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--check-collection',
        action='store_true',
        help='Check collection status before processing'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of iterations for the agent (optional)'
    )
    
    parser.add_argument(
        '--debug-retrieval',
        action='store_true',
        help='Enable detailed debugging for document retrieval'
    )
    
    return parser.parse_args()

def print_result(result: Dict[str, Any], verbose: bool = False):
    """Pretty print the result to the console."""
    print("\n" + "="*80)
    print("DOCUMENT RESEARCH AGENT RESULT")
    print("="*80)
    
    # Print status
    if not result.get("success", True):
        print("\nSTATUS: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("\nSTATUS: SUCCESS")
        if "iterations" in result:
            print(f"Iterations: {result['iterations']}")
    
    print("\nANSWER:")
    print(result.get("final_answer", "No answer generated"))
    
    print("\nCITATIONS:")
    citations = result.get("citations", [])
    if citations:
        for i, citation in enumerate(citations):
            print(f"\n[{i+1}] Source: {citation.get('filename', 'unknown')}, Page: {citation.get('page', 0)}")
            print(f"Text: {citation.get('text', '')}")
    else:
        print("No citations provided.")
    
    # Print token usage information if available
    if "token_usage" in result:
        token_usage = result["token_usage"]
        print("\nTOKEN USAGE:")
        print(f"- Prompt tokens: {token_usage.get('prompt_tokens', 0)}")
        print(f"- Completion tokens: {token_usage.get('completion_tokens', 0)}")
        print(f"- Total tokens: {token_usage.get('total_tokens', 0)}")
    
    if verbose and "agent_scratchpad" in result:
        print("\n" + "="*80)
        print("AGENT REASONING (DEBUG):")
        print("="*80)
        print(result.get("agent_scratchpad", "No reasoning available"))
    
    print("\n" + "="*80)

def save_to_file(result: Dict[str, Any], filepath: str):
    """Save the result to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving result to file: {e}")

def check_document_availability(filenames: List[str]) -> bool:
    """Check if the specified documents exist in the test_data directory."""
    test_data_dir = "test_data"
    all_found = True
    
    for filename in filenames:
        # Check if the file exists in test_data directory
        file_path = os.path.join(test_data_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Document not found in test_data directory: {filename}")
            all_found = False
    
    return all_found

def main():
    """Main entry point for the Document Research Agent."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Enable debug logging for document retrieval if requested
    if args.debug_retrieval:
        logging.getLogger('src.retriever').setLevel(logging.DEBUG)
        logger.info("Document retrieval debugging enabled")
    
    try:
        # Check if documents exist
        if not check_document_availability(args.filenames):
            logger.warning("Some specified documents were not found. Results may be affected.")
        
        # Initialize the agent
        agent = DocumentResearchAgent()
        
        # Check collection status if requested
        if args.check_collection:
            logger.info("Checking Chroma DB collection status...")
            status = agent.check_collection_status(args.filenames)
            
            if status["success"]:
                print("\nCollection check successful:")
                print(f"- Document count: {status['document_count']}")
                if status.get("filenames_found") is not None:
                    print(f"- Specified filenames found: {'Yes' if status['filenames_found'] else 'No'}")
                logger.info("Collection check completed successfully.")
                return 0
            else:
                print("\nCollection check failed:")
                print(f"- Error: {status.get('error', 'Unknown error')}")
                print(f"- Document count: {status.get('document_count', 0)}")
                if status.get("filenames_found") is not None:
                    print(f"- Specified filenames found: {'Yes' if status['filenames_found'] else 'No'}")
                logger.error("Collection check failed.")
                return 1
        
        # Run the agent
        logger.info(f"Starting research with query: {args.query}")
        result = agent.run(
            query=args.query,
            filenames=args.filenames,
            max_iterations=args.max_iterations,
            include_scratchpad=args.verbose
        )
        
        # Print the result
        print_result(result, args.verbose)
        
        # Save to file if requested
        if args.output:
            save_to_file(result, args.output)
        
        # Return appropriate exit code
        return 0 if result.get("success", False) else 1
    
    except Exception as e:
        logger.error(f"Error running Document Research Agent: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 