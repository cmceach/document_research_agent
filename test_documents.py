#!/usr/bin/env python3
"""
Test script for running sample queries against the Document Research Agent.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Dict, Any

# Sample queries to test
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

def run_query(query: str, filenames: List[str], verbose: bool = False) -> None:
    """Run a query against the Document Research Agent."""
    # Build the command
    cmd = ["python", "-m", "src.main", query, "--filenames"]
    cmd.extend(filenames)
    
    if verbose:
        cmd.append("--verbose")
    
    # Print the command
    print("\n" + "=" * 80)
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

def main():
    """Run sample queries against the Document Research Agent."""
    parser = argparse.ArgumentParser(description="Test Document Research Agent with sample queries")
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--query-index", 
        type=int, 
        help="Run only a specific query by index (0-based)"
    )
    parser.add_argument(
        "--check-collection",
        action="store_true",
        help="Check collection status for each query instead of running it"
    )
    args = parser.parse_args()
    
    # Check if we should run a specific query
    if args.query_index is not None:
        if 0 <= args.query_index < len(SAMPLE_QUERIES):
            query_info = SAMPLE_QUERIES[args.query_index]
            
            # Modify command if we're just checking collection status
            if args.check_collection:
                cmd = ["python", "-m", "src.main", query_info["query"], "--filenames"]
                cmd.extend(query_info["filenames"])
                cmd.append("--check-collection")
                
                print(f"Checking collection status for query {args.query_index}...")
                subprocess.run(cmd, check=True)
            else:
                run_query(query_info["query"], query_info["filenames"], args.verbose)
        else:
            print(f"Invalid query index. Must be between 0 and {len(SAMPLE_QUERIES)-1}")
            return 1
    else:
        # Run all queries
        for i, query_info in enumerate(SAMPLE_QUERIES):
            print(f"\nRunning query {i}...")
            
            # Modify command if we're just checking collection status
            if args.check_collection:
                cmd = ["python", "-m", "src.main", query_info["query"], "--filenames"]
                cmd.extend(query_info["filenames"])
                cmd.append("--check-collection")
                
                print(f"Checking collection status for query {i}...")
                subprocess.run(cmd, check=True)
            else:
                run_query(query_info["query"], query_info["filenames"], args.verbose)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 