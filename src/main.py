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
    parser = argparse.ArgumentParser(description='Document Research Agent or FastAPI Server')
    
    # Arguments for running as a command-line tool
    tool_group = parser.add_argument_group('Command-line Tool Options')
    tool_group.add_argument(
        'query',
        type=str,
        nargs='?', # Make query optional if --serve is used
        default=None, # Default to None if not provided
        help='The question to research in the documents (required if not using --serve)'
    )
    
    tool_group.add_argument(
        '--filenames',
        type=str,
        nargs='+',
        # required=True, # No longer always required
        help='List of document filenames to search within (required if not using --serve)'
    )
    
    tool_group.add_argument(
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

    # Argument for running as a FastAPI server
    server_group = parser.add_argument_group('FastAPI Server Options')
    server_group.add_argument(
        '--serve',
        action='store_true',
        help='Run the FastAPI server'
    )
    server_group.add_argument(
        '--host',
        type=str,
        default="0.0.0.0",
        help='Host for the FastAPI server (default: 0.0.0.0)'
    )
    server_group.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the FastAPI server (default: 8000)'
    )
    
    args = parser.parse_args()

    # Validate arguments: query and filenames are required if --serve is not used
    if not args.serve and (args.query is None or args.filenames is None):
        parser.error("the following arguments are required when not using --serve: query, --filenames")

    return args

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
    
    # Print search queries by iteration
    search_queries_by_iteration = result.get("search_queries_by_iteration", [])
    if search_queries_by_iteration:
        print("\nSEARCH QUERIES BY ITERATION:")
        for query_info in search_queries_by_iteration:
            iteration = query_info.get("iteration", "?")
            attempt = query_info.get("attempt", "?")
            queries = query_info.get("queries", [])
            context_items = query_info.get("context_items_available", 0)
            
            print(f"\nIteration {iteration}.{attempt} (Context items available: {context_items}):")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. \"{query}\"")
    
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
    
    # Print runtime information if available
    if "runtime" in result:
        runtime = result["runtime"]
        print("\nRUNTIME:")
        print(f"- Start time: {runtime.get('start_time', 'unknown')}")
        print(f"- End time: {runtime.get('end_time', 'unknown')}")
        print(f"- Total runtime: {runtime.get('runtime_formatted', 'unknown')}")
        print(f"- Total seconds: {runtime.get('runtime_seconds', 0):.2f}")
    
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

    if args.serve:
        try:
            import uvicorn
            from src.api import app  # Import the FastAPI app
            logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
            return 0 # Should not be reached if uvicorn runs successfully
        except ImportError:
            logger.error("uvicorn is not installed. Please install it with `pip install uvicorn` to use --serve.")
            return 1
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}", exc_info=True)
            return 1

    # Original command-line tool logic starts here
    if args.query is None or args.filenames is None:
        logger.error("Query and filenames must be provided when not running the server.")
        # parse_arguments already handles this, but as a safeguard:
        return 1

    # Enable debug logging for document retrieval if requested
    if args.debug_retrieval:
        logging.getLogger('src.retriever').setLevel(logging.DEBUG)
        logger.info("Document retrieval debugging enabled")
    
    try:
        # Check if documents exist - only if filenames are provided
        if args.filenames and not check_document_availability(args.filenames):
            logger.warning("Some specified documents were not found. Results may be affected.")
        
        # Initialize the agent - only if query and filenames are provided for CLI mode
        agent = DocumentResearchAgent() # This will be initialized only if not serving

        if args.check_collection:
            if not args.filenames:
                logger.error("--check-collection requires --filenames to be specified.")
                return 1
            logger.info("Checking Chroma DB collection status...")
            status = agent.check_collection_status(args.filenames)
            
            if status["success"]:
                print("\nCollection check successful:")
                print(f"- Document count: {status['document_count']}")
                if status.get("filenames_found") is not None:
                    print(f"- Specified filenames found: {'Yes' if status['filenames_found'] else 'No'}")
                logger.info("Collection check completed successfully.")
                return 0 # Exit after checking collection
            else:
                print("\nCollection check failed:")
                print(f"- Error: {status.get('error', 'Unknown error')}")
                print(f"- Document count: {status.get('document_count', 0)}")
                if status.get("filenames_found") is not None:
                    print(f"- Specified filenames found: {'Yes' if status['filenames_found'] else 'No'}")
                logger.error("Collection check failed.")
                return 1 # Exit after checking collection
        
        # Proceed to run the agent only if not serving and not just checking collection
        if args.query and args.filenames: # Ensure query and filenames are present for CLI run
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
        elif not args.serve: # If not serving and query/filenames are missing (should be caught by parser)
            logger.error("Query and filenames are required for command-line execution.")
            return 1
        
        return 0 # Default exit for non-error cases not explicitly handled (e.g. if --serve was not set but no other action)

    except Exception as e:
        # Avoid logging "NoneType object has no attribute 'filenames'" if --serve was used
        if not args.serve:
            logger.error(f"Error running Document Research Agent: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 