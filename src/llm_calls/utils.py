"""
Utility functions for LLM calls in the Document Research Agent.
"""

from typing import Dict, List, Any, Optional

def format_context(retrieved_context: List[Dict[str, Any]], max_items: int = 15, max_chars: int = 300) -> str:
    """
    Format context items efficiently for LLM prompts with token optimization.
    
    Args:
        retrieved_context: List of context items with text, filename, and page
        max_items: Maximum number of context items to include
        max_chars: Maximum characters per context item
        
    Returns:
        Formatted context string optimized for token usage
    """
    if not retrieved_context:
        return "None"
    
    # Limit number of items to reduce token usage
    items = retrieved_context[-max_items:] if len(retrieved_context) > max_items else retrieved_context
    
    formatted_parts = []
    for i, item in enumerate(items):
        # Truncate text to reduce tokens
        text = item['text'][:max_chars] + "..." if len(item['text']) > max_chars else item['text']
        # Compact format to save tokens
        formatted_parts.append(f"[{i+1}] {text} (p{item['page']}, {item['filename']})")
    
    return "\n".join(formatted_parts)

def format_previous_queries(previous_queries: List[str]) -> str:
    """
    Format previous search queries efficiently.
    
    Args:
        previous_queries: List of query strings
        
    Returns:
        Compact formatted string
    """
    if not previous_queries:
        return "None"
    
    # Limit to last 5 queries to save tokens
    recent_queries = previous_queries[-5:] if len(previous_queries) > 5 else previous_queries
    return ", ".join(f'"{q}"' for q in recent_queries)

def clean_query_results(queries: List[str], max_length: int = 80, max_queries: int = 3) -> List[str]:
    """
    Clean and normalize query results with stricter limits for token efficiency.
    
    Args:
        queries: List of raw query strings
        max_length: Maximum length of each query (reduced from 100)
        max_queries: Maximum number of queries to return
        
    Returns:
        Cleaned list of query strings
    """
    # Filter out any empty queries and strip whitespace
    queries = [q.strip() for q in queries if q and q.strip()]
    
    # Limit length of each query more aggressively
    queries = [q[:max_length] for q in queries]
    
    # Limit number of queries
    return queries[:max_queries]

def truncate_context_for_tokens(
    context_list: List[Dict[str, Any]], 
    max_items: int = 15,  # Reduced default from None
    max_chars_per_item: int = 400,  # Reduced default
    preserve_recent: bool = True
) -> List[Dict[str, Any]]:
    """
    Aggressively truncate context items to minimize token usage.
    
    Args:
        context_list: List of context items to truncate
        max_items: Maximum number of context items (default 15)
        max_chars_per_item: Maximum characters per item (default 400)
        preserve_recent: Whether to prioritize recent context items
        
    Returns:
        Truncated list of context items
    """
    if not context_list:
        return []
    
    # Create a copy to avoid modifying the original
    result = context_list.copy()
    
    # Apply max_items limit
    if len(result) > max_items:
        if preserve_recent:
            result = result[-max_items:]
        else:
            result = result[:max_items]
    
    # Apply character limit per item
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
    # Improved approximation: 1 token ≈ 3.5 characters for English text
    return int(len(text) / 3.5)

def optimize_context_for_prompt(
    context_list: List[Dict[str, Any]], 
    target_tokens: int = 8000
) -> List[Dict[str, Any]]:
    """
    Optimize context list to fit within target token count.
    
    Args:
        context_list: List of context items
        target_tokens: Target token count for the context
        
    Returns:
        Optimized context list that fits within token limit
    """
    if not context_list:
        return []
    
    # Start with recent items and work backwards
    optimized = []
    current_tokens = 0
    
    for item in reversed(context_list):
        item_tokens = estimate_token_count(item["text"])
        
        if current_tokens + item_tokens <= target_tokens:
            optimized.insert(0, item)  # Insert at beginning to maintain order
            current_tokens += item_tokens
        else:
            # Try to fit a truncated version
            remaining_tokens = target_tokens - current_tokens
            if remaining_tokens > 100:  # Only if we have reasonable space left
                truncated_chars = int(remaining_tokens * 3.5 * 0.8)  # Leave some buffer
                truncated_item = item.copy()
                truncated_item["text"] = item["text"][:truncated_chars] + "..."
                optimized.insert(0, truncated_item)
            break
    
    return optimized

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
        # Use first 200 chars for deduplication to be more efficient
        content_key = content[:200] if content else ""
        if content_key and content_key not in unique_contents:
            unique_contents.add(content_key)
            deduplicated_results.append(result)
            
    return deduplicated_results 