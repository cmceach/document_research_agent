#!/usr/bin/env python
"""Test script for the LangChain structured output integration."""
import os
import sys
from dotenv import load_dotenv
from src.llm_calls.llm_wrappers_langchain import LLMWrappers

def main():
    """Run a test of the LangChain implementation with actual LLM calls."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set model to GPT-4o
    os.environ["OPENAI_CHAT_MODEL_NAME"] = "gpt-4o"
    
    print(f"Starting LangChain structured output integration test with model: {os.environ['OPENAI_CHAT_MODEL_NAME']}...")
    
    # Initialize the LLM wrapper
    llm_wrapper = LLMWrappers()
    
    try:
        # Test 1: Generate search queries
        print("\n--- Testing search query generation ---")
        original_query = "What are the latest advancements in quantum computing?"
        previous_queries = []
        context = []
        
        print(f"Original query: {original_query}")
        search_queries = llm_wrapper.generate_search_queries_llm(
            original_query=original_query,
            retrieved_context=context,
            previous_queries=previous_queries,
            generation_attempt=1
        )
        
        print("Generated search queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")
        
        # Test 2: Grade context
        print("\n--- Testing context grading ---")
        mock_context = [
            {
                "text": "Quantum computing has seen significant advancements in recent years. IBM unveiled a 127-qubit quantum processor in 2021, demonstrating progress in scaling quantum systems.",
                "page": 1,
                "filename": "quantum_advances.pdf"
            },
            {
                "text": "Google claimed quantum supremacy in 2019 with their 53-qubit Sycamore processor, performing a calculation that would take classical supercomputers thousands of years.",
                "page": 2,
                "filename": "quantum_advances.pdf"
            }
        ]
        
        print("Context to grade:")
        for i, ctx in enumerate(mock_context, 1):
            print(f"  {i}. {ctx['text']}")
        
        decision = llm_wrapper.grade_context_llm(
            original_query=original_query,
            retrieved_context=mock_context,
            iterations=1,
            max_iterations=3
        )
        
        print(f"Grading decision: {decision}")
        
        # Test 3: Generate final answer
        print("\n--- Testing final answer generation ---")
        
        answer, citations = llm_wrapper.generate_final_answer_llm(
            original_query=original_query,
            retrieved_context=mock_context
        )
        
        print("Final answer:")
        print(answer)
        
        print("\nCitations:")
        for i, citation in enumerate(citations, 1):
            print(f"  {i}. {citation['text']} (Page {citation['page']} in {citation['filename']})")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 