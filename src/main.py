import os
import json
import argparse
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.graph_builder import build_graph
from src.retriever.chroma_retriever import ChromaRetriever

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
    
    return parser.parse_args()

def format_output(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Format the final state into the expected output format."""
    return {
        "answer": final_state.get("final_answer", "No answer generated"),
        "citations": final_state.get("citations", [])
    }

def print_result(result: Dict[str, Any], verbose: bool = False):
    """Pretty print the result to the console."""
    print("\n" + "="*80)
    print("DOCUMENT RESEARCH AGENT RESULT")
    print("="*80)
    
    print("\nANSWER:")
    print(result["answer"])
    
    print("\nCITATIONS:")
    if result["citations"]:
        for i, citation in enumerate(result["citations"]):
            print(f"\n[{i+1}] Source: {citation.get('filename', 'unknown')}, Page: {citation.get('page', 0)}")
            print(f"Text: {citation.get('text', '')}")
    else:
        print("No citations provided.")
    
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

def check_environment():
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

def check_collection_status(filenames: List[str]) -> bool:
    """Check if the Chroma DB collection exists and has documents for the specified filenames."""
    try:
        # Initialize the retriever
        retriever = ChromaRetriever()
        
        # Get collection statistics
        stats = retriever.get_collection_stats()
        
        if stats["document_count"] == 0:
            logger.error("Collection exists but contains no documents.")
            return False
        
        # Test retrieval with a basic query to check for any documents with the specified filenames
        test_query = "test query"  # Not important, just checking if any documents match filenames
        results = retriever.retrieve_context(
            search_queries=[test_query],
            filenames=filenames,
            top_k=1
        )
        
        if not results:
            logger.warning(f"No documents found matching the specified filenames: {filenames}")
            logger.warning("Please check that the documents are properly loaded into Chroma DB.")
            return False
        
        logger.info(f"Collection check successful: {stats['document_count']} total documents, found matching documents for filenames.")
        return True
        
    except Exception as e:
        logger.error(f"Error checking collection status: {e}")
        return False

def main():
    """Main entry point for the Document Research Agent."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check environment
    if not check_environment():
        return 1
    
    # Check collection status if requested
    if args.check_collection:
        logger.info("Checking Chroma DB collection status...")
        if not check_collection_status(args.filenames):
            logger.error("Collection check failed. Please ensure your documents are properly loaded.")
            return 1
        logger.info("Collection check completed successfully.")
        return 0
    
    try:
        # Build the graph
        graph = build_graph()
        
        # Prepare the input state
        input_state = {
            "original_query": args.query,
            "filenames": args.filenames
        }
        
        logger.info(f"Starting research with query: {args.query}")
        logger.info(f"Document filenames: {args.filenames}")
        
        # Execute the graph
        final_state = graph.invoke(input_state)
        
        # Format the output
        result = format_output(final_state)
        
        # Print the result
        print_result(result, args.verbose)
        
        # Save to file if requested
        if args.output:
            save_to_file(result, args.output)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running Document Research Agent: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 