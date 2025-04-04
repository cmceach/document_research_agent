from typing import Dict, List, Optional, TypedDict, Any, Annotated
import operator

class AgentState(TypedDict):
    """State for the Document Research Agent."""
    
    # Input parameters
    original_query: str
    filenames: List[str]
    
    # Iteration tracking
    iterations: int
    max_iterations: int
    generation_attempts: int
    max_generation_attempts: int
    
    # Context management
    search_queries: List[str]
    # Using Annotated with operator.add for lists that should be appended to, not replaced
    retrieved_context: Annotated[List[Dict[str, Any]], operator.add]  # List of {"text": str, "page": int, "filename": str}
    
    # Agent's working memory
    agent_scratchpad: str
    
    # Output fields
    final_answer: Optional[str]
    citations: Annotated[List[Dict[str, Any]], operator.add]  # List of {"text": str, "page": int, "filename": str}
    is_finished: bool 