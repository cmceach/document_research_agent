from textwrap import dedent

# Prompts for generating search queries - Optimized for token efficiency
SEARCH_QUERIES_SYSTEM_PROMPT = dedent("""
    Generate 2-3 targeted search queries to find information in documents.
    Make queries specific, use synonyms, and focus on different aspects of the question.
    """).strip()

SEARCH_QUERIES_USER_PROMPT = dedent("""
    Question: {original_query}
    Previous queries: {formatted_prev_queries}
    Context so far: {formatted_context}
    
    Generate 2-3 search queries (attempt #{generation_attempt}):
    """).strip()

# Prompts for grading context - Optimized for token efficiency
GRADE_CONTEXT_SYSTEM_PROMPT = dedent("""
    Evaluate if context is sufficient to answer the question.
    Return:
    - FINISH: Can answer completely
    - CONTINUE: Need more context
    - RETRY_GENERATION: Try different search strategy
    - FAIL: Information likely not available
    """).strip()

GRADE_CONTEXT_USER_PROMPT = dedent("""
    Question: {original_query}
    Iteration: {iterations}/{max_iterations}
    Context: {formatted_context}
    
    Decision (FINISH/CONTINUE/RETRY_GENERATION/FAIL):
    """).strip()

# Prompts for generating final answer - Optimized for token efficiency
FINAL_ANSWER_SYSTEM_PROMPT = dedent("""
    Answer the question using ONLY the provided context.
    Include citations [X] for each claim.
    If insufficient information, state clearly.
    """).strip()

FINAL_ANSWER_USER_PROMPT = dedent("""
    Question: {original_query}
    Context: {formatted_context}
    
    Answer with citations:
    """).strip() 