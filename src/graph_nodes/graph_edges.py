from typing import Literal
import logging

from src.graph_state.agent_state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def should_continue(state: AgentState) -> Literal["generate_search_queries", "generate_final_answer", "handle_failure"]:
    """
    Determines the next node to call based on the context evaluation and iteration state.
    
    Args:
        state: Current agent state
        
    Returns:
        String indicating the next node to call
    """
    decision = state.get("context_decision", "CONTINUE")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 5)
    generation_attempts = state.get("generation_attempts", 0)
    max_generation_attempts = state.get("max_generation_attempts", 3)
    
    logger.info(f"Edge decision: {decision}, Iterations: {iterations}/{max_iterations}, " 
               f"Generation attempts: {generation_attempts}/{max_generation_attempts}")
    
    # If we've reached the maximum iterations, we should finish
    if iterations >= max_iterations:
        if state.get("retrieved_context"):
            return "generate_final_answer"  # We have some context, try to answer
        else:
            return "handle_failure"  # No context at all after max iterations
    
    # Check decision from context grading
    if decision == "FINISH":
        return "generate_final_answer"
    elif decision == "FAIL":
        return "handle_failure"
    elif decision == "RETRY_GENERATION" or decision == "CONTINUE":
        # Check if we've exceeded max generation attempts for this iteration
        if generation_attempts >= max_generation_attempts:
            # If we've tried too many times without success but have some context
            if state.get("retrieved_context"):
                return "generate_final_answer"  # Try to answer with what we have
            else:
                return "handle_failure"  # No useful context after many attempts
        else:
            return "generate_search_queries"  # Try again with new queries
    
    # Default fallback - try to generate new search queries
    return "generate_search_queries" 