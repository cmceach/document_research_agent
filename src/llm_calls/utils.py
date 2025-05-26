"""
Utility functions for LLM calls in the Document Research Agent.
"""

from typing import Dict, List, Any, Optional

def format_context(retrieved_context: List[Dict[str, Any]]) -> str:
    """
    Format a list of context items into a readable string for LLM prompts.
    
    Args:
        retrieved_context: List of context items with text, filename, and page
        
    Returns:
        Formatted context string with numbered items
    """
    if not retrieved_context:
        return "No context retrieved yet."
        
    formatted_context = ""
    for i, item in enumerate(retrieved_context):
        formatted_context += f"[{i+1}] Text: {item['text']}\n"
        formatted_context += f"Source: {item['filename']}, Page: {item['page']}\n\n"
    
    return formatted_context.strip()

def format_previous_queries(previous_queries: List[str]) -> str:
    """
    Format a list of previous search queries into a readable string.
    
    Args:
        previous_queries: List of query strings
        
    Returns:
        Formatted string with bulleted query items
    """
    if not previous_queries:
        return "None"
        
    return "\n".join([f"- {q}" for q in previous_queries])

def clean_query_results(queries: List[str], max_length: int = 100, max_queries: int = 3) -> List[str]:
    """
    Clean and normalize query results by:
    - Removing empty queries
    - Trimming to max length
    - Limiting number of queries
    
    Args:
        queries: List of raw query strings
        max_length: Maximum length of each query
        max_queries: Maximum number of queries to return
        
    Returns:
        Cleaned list of query strings
    """
    # Filter out any empty queries and strip whitespace
    queries = [q.strip() for q in queries if q and q.strip()]
    
    # Limit length of each query
    queries = [q[:max_length] for q in queries]
    
    # Limit number of queries
    return queries[:max_queries]

def truncate_context_for_tokens(
    context_list: List[Dict[str, Any]], 
    max_items: Optional[int] = None,
    max_chars_per_item: Optional[int] = None,
    preserve_recent: bool = True
) -> List[Dict[str, Any]]:
    """
    Truncate context items to stay within token limits.
    
    Args:
        context_list: List of context items to truncate
        max_items: Maximum number of context items to include
        max_chars_per_item: Maximum characters per context item text field
        preserve_recent: Whether to prioritize keeping the most recent context items
        
    Returns:
        Truncated list of context items
    """
    if not context_list:
        return []
    
    # Create a copy to avoid modifying the original
    result = context_list.copy()
    
    # Apply max_items limit if specified
    if max_items is not None and len(result) > max_items:
        # If preserving recent items, take the last max_items
        if preserve_recent:
            result = result[-max_items:]
        else:
            result = result[:max_items]
    
    # Apply character limit per item if specified
    if max_chars_per_item is not None:
        for item in result:
            if len(item["text"]) > max_chars_per_item:
                item["text"] = item["text"][:max_chars_per_item] + "..."
    
    return result

def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a string. This is a rough approximation.
    
    Args:
        text: String to estimate token count for
        
    Returns:
        Estimated token count
    """
    # Very rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def deduplicate_search_results(results: List[Dict[str, Any]], existing_contents: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Deduplicate search results based on content.
    
    Args:
        results: List of search result dictionaries
        existing_contents: Optional set of already seen content
        
    Returns:
        List of deduplicated search results
    """
    unique_contents = existing_contents or set()
    deduplicated_results = []
    
    for result in results:
        content = result.get("text", "")
        if content and content not in unique_contents:
            unique_contents.add(content)
            deduplicated_results.append(result)
            
    return deduplicated_results 