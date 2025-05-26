from typing import Dict, List, Optional, TypedDict, Any, Annotated
import operator
from src.llm_calls.llm_wrappers import LLMWrappers

class AgentState(TypedDict):
    """State for the Document Research Agent."""
    
    # Input parameters
    original_query: str
    filenames: List[str]
    llm_wrapper: LLMWrappers  # LLM wrapper instance for token tracking
    
    # Iteration tracking
    iterations: int
    max_iterations: int
    generation_attempts: int
    max_generation_attempts: int
    
    # Context management
    search_queries: List[str]
    search_queries_by_iteration: Annotated[List[Dict[str, Any]], operator.add]  # Track queries per iteration
    retrieved_context: Annotated[List[Dict[str, Any]], operator.add]  # List of {"text": str, "page": int, "filename": str}
    
    # Agent's working memory
    agent_scratchpad: str
    
    # Output fields
    final_answer: Optional[str]
    citations: Annotated[List[Dict[str, Any]], operator.add]  # List of {"text": str, "page": int, "filename": str}
    is_finished: bool 