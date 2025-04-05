#!/usr/bin/env python3
"""
Script to switch between the original OpenAI implementation and the LangChain implementation of the Document Research Agent.
"""

import os
import re
import shutil
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of the given file"""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    return backup_path

def update_node_functions(target_implementation):
    """Update node_functions.py to use the specified implementation"""
    file_path = "src/graph_nodes/node_functions.py"
    
    # Create backup
    backup_file(file_path)
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Update the import statement
    if target_implementation == "langchain":
        updated_content = content.replace(
            "from src.llm_calls.llm_wrappers import LLMWrappers",
            "from src.llm_calls.llm_wrappers_langchain import LLMWrappers"
        )
        logger.info("Switching to LangChain implementation")
    else:  # openai
        updated_content = content.replace(
            "from src.llm_calls.llm_wrappers_langchain import LLMWrappers",
            "from src.llm_calls.llm_wrappers import LLMWrappers"
        )
        logger.info("Switching to original OpenAI implementation")
    
    # Write the updated content
    with open(file_path, 'w') as file:
        file.write(updated_content)
    
    logger.info(f"Updated {file_path} to use {target_implementation} implementation")

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Switch LLM implementation for Document Research Agent')
    parser.add_argument('implementation', choices=['openai', 'langchain'], 
                       help='The implementation to use: "openai" for original implementation or "langchain" for LangChain implementation')
    
    args = parser.parse_args()
    
    logger.info(f"Switching to {args.implementation} implementation")
    
    try:
        update_node_functions(args.implementation)
        logger.info(f"Successfully switched to {args.implementation} implementation")
        logger.info("To revert changes, restore the .bak files")
    except Exception as e:
        logger.error(f"Error during update: {e}")
        logger.info("Please restore from backups if needed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 