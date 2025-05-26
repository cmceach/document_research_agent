from typing import Literal, Dict, Any, Tuple
import logging

from src.graph_state.agent_state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_state_limits(state: AgentState) -> Tuple[bool, bool]:
    """
    Validate iteration and generation attempt limits.
    
    Args:
        state: Current agent state
        
    Returns:
        Tuple of (reached_max_iterations, reached_max_attempts)
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 5)
    generation_attempts = state.get("generation_attempts", 0)
    max_generation_attempts = state.get("max_generation_attempts", 3)
    
    return (
        iterations >= max_iterations,
        generation_attempts >= max_generation_attempts
    )

def should_continue(state: AgentState) -> Literal["generate_search_queries", "generate_final_answer", "handle_failure"]:
    """
    Determines the next node to call based on the context evaluation and iteration state.
    
    Args:
        state: Current agent state
        
    Returns:
        String indicating the next node to call
    """
    decision = state.get("context_decision", "CONTINUE")
    reached_max_iterations, reached_max_attempts = validate_state_limits(state)
    
    logger.info(f"Edge decision: {decision}, Max iterations reached: {reached_max_iterations}, " 
               f"Max attempts reached: {reached_max_attempts}")
    
    # If we've reached the maximum iterations
    if reached_max_iterations:
        if state.get("retrieved_context"):
            return "generate_final_answer"  # We have some context, try to answer
        else:
            return "handle_failure"  # No context at all after max iterations
            
    # Handle other decision cases
    if decision == "FINISH":
        return "generate_final_answer"
    elif decision == "RETRY_GENERATION" and not reached_max_attempts:
        return "generate_search_queries"
    elif decision == "FAIL":
        return "handle_failure"
    else:
        return "generate_search_queries"  # Default to continuing the search 