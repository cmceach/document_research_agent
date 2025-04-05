from typing import Dict, Any
import logging

from src.graph_state.agent_state import AgentState
from src.retriever.chroma_retriever import ChromaRetriever
from src.llm_calls.llm_wrappers import LLMWrappers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use lazy initialization for clients to make testing easier
retriever = ChromaRetriever(lazy_init=True)
llm_wrapper = LLMWrappers(lazy_init=True)

def start_node_func(state: AgentState) -> Dict[str, Any]:
    """Initialize the agent state with default values."""
    logger.info(f"Starting agent with query: {state['original_query']}")
    
    # Return state delta with initial values
    return {
        "iterations": 0,
        "max_iterations": 5,
        "generation_attempts": 0,
        "max_generation_attempts": 3,
        "search_queries": [],
        "retrieved_context": [],
        "agent_scratchpad": "",
        "final_answer": None,
        "citations": [],
        "is_finished": False
    }

def generate_search_queries_node_func(state: AgentState) -> Dict[str, Any]:
    """Generate search queries based on the original query and current context."""
    logger.info(f"Generating search queries. Iteration: {state['iterations']}, Generation attempt: {state['generation_attempts']}")
    
    # Generate search queries
    queries = llm_wrapper.generate_search_queries_llm(
        original_query=state["original_query"],
        retrieved_context=state["retrieved_context"],
        previous_queries=state["search_queries"],
        generation_attempt=state["generation_attempts"] + 1
    )
    
    # Update the scratchpad
    scratchpad = state.get("agent_scratchpad", "")
    scratchpad += f"\n\nGenerated search queries (attempt {state['generation_attempts'] + 1}):\n"
    scratchpad += "\n".join([f"- {q}" for q in queries])
    
    # Return state delta
    return {
        "search_queries": queries,
        "generation_attempts": state["generation_attempts"] + 1,
        "agent_scratchpad": scratchpad
    }

def retrieve_context_node_func(state: AgentState) -> Dict[str, Any]:
    """Retrieve context from Chroma DB using the generated queries."""
    logger.info(f"Retrieving context. Iteration: {state['iterations']}")
    
    # Execute search queries
    new_context = retriever.retrieve_context(
        search_queries=state["search_queries"],
        filenames=state["filenames"]
    )
    
    # Update the scratchpad
    scratchpad = state.get("agent_scratchpad", "")
    scratchpad += f"\n\nRetrieved {len(new_context)} new context items in iteration {state['iterations'] + 1}:\n"
    
    if new_context:
        for i, item in enumerate(new_context):
            scratchpad += f"[{i+1}] Source: {item['filename']}, Page: {item['page']}\n"
            scratchpad += f"Text: {item['text'][:150]}...\n\n"
    else:
        scratchpad += "No new relevant context found.\n"
    
    # With Annotated[List, operator.add] in the state, we only need to return the new context
    # and it will be appended to the existing list
    return {
        "retrieved_context": new_context,  # Now appended due to annotation in state
        "iterations": state["iterations"] + 1,
        "agent_scratchpad": scratchpad
    }

def grade_context_node_func(state: AgentState) -> Dict[str, Any]:
    """Evaluate if the context is sufficient to answer the query."""
    logger.info(f"Grading context. Iteration: {state['iterations']}")
    
    # Grade the context
    decision = llm_wrapper.grade_context_llm(
        original_query=state["original_query"],
        retrieved_context=state["retrieved_context"],
        iterations=state["iterations"],
        max_iterations=state["max_iterations"]
    )
    
    # Update the scratchpad
    scratchpad = state.get("agent_scratchpad", "")
    scratchpad += f"\n\nContext evaluation decision: {decision}\n"
    
    # Reset generation attempts if we're continuing with a new search strategy
    generation_attempts = state["generation_attempts"]
    if decision == "RETRY_GENERATION":
        generation_attempts = 0
        scratchpad += "Resetting search strategy for next iteration.\n"
    
    # Return state delta
    return {
        "agent_scratchpad": scratchpad,
        "generation_attempts": generation_attempts,
        "context_decision": decision  # This will be used by edge logic but doesn't need to be part of the formal state
    }

def generate_final_answer_node_func(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer based on the retrieved context."""
    logger.info("Generating final answer")
    
    # Generate the answer
    answer, citations = llm_wrapper.generate_final_answer_llm(
        original_query=state["original_query"],
        retrieved_context=state["retrieved_context"]
    )
    
    # Update the scratchpad
    scratchpad = state.get("agent_scratchpad", "")
    scratchpad += "\n\nGenerating final answer...\n"
    
    # Return state delta - citations will be appended due to annotation
    return {
        "final_answer": answer,
        "citations": citations,
        "is_finished": True,
        "agent_scratchpad": scratchpad
    }

def handle_failure_node_func(state: AgentState) -> Dict[str, Any]:
    """Handle the case where no relevant information was found."""
    logger.info("Handling failure case - no relevant information found")
    
    # Set default failure response
    answer = "Information not found in provided documents"
    
    # Update the scratchpad
    scratchpad = state.get("agent_scratchpad", "")
    scratchpad += "\n\nNo relevant information found in the provided documents.\n"
    
    # Return state delta
    return {
        "final_answer": answer,
        "citations": [],  # Empty list, no append needed
        "is_finished": True,
        "agent_scratchpad": scratchpad
    } 