import logging
from langgraph.graph import StateGraph, END
from typing import Dict, Any

from src.graph_state.agent_state import AgentState
from src.graph_nodes.node_functions import (
    start_node_func,
    generate_search_queries_node_func,
    retrieve_context_node_func,
    grade_context_node_func,
    generate_final_answer_node_func,
    handle_failure_node_func
)
from src.graph_nodes.graph_edges import should_continue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow for the Document Research Agent.
    
    Returns:
        StateGraph: The compiled graph ready for execution
    """
    logger.info("Building Document Research Agent graph")
    
    # Initialize the graph with our state type
    workflow = StateGraph(AgentState)
    
    # Add all nodes - StateGraph in newer versions requires more explicit mapping
    workflow.add_node("start", start_node_func)
    workflow.add_node("generate_search_queries", generate_search_queries_node_func)
    workflow.add_node("retrieve_context", retrieve_context_node_func)
    workflow.add_node("grade_context", grade_context_node_func)
    workflow.add_node("generate_final_answer", generate_final_answer_node_func)
    workflow.add_node("handle_failure", handle_failure_node_func)
    
    # Set entry point
    workflow.set_entry_point("start")
    
    # Add static edges
    workflow.add_edge("start", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "retrieve_context")
    workflow.add_edge("retrieve_context", "grade_context")
    
    # Add conditional edge from grading node
    workflow.add_conditional_edges(
        "grade_context",
        should_continue,
        {
            "generate_search_queries": "generate_search_queries",
            "generate_final_answer": "generate_final_answer",
            "handle_failure": "handle_failure"
        }
    )
    
    # Final nodes lead to end
    workflow.add_edge("generate_final_answer", END)
    workflow.add_edge("handle_failure", END)
    
    # Compile the graph - use updated compile method
    graph = workflow.compile()
    logger.info("Document Research Agent graph built and compiled successfully")
    
    return graph 