from textwrap import dedent

# Prompts for generating search queries
SEARCH_QUERIES_SYSTEM_PROMPT = dedent("""
    You are a skilled researcher with the ability to formulate targeted search queries to find information in a document.
    Based on the original question and any previously retrieved context, generate specific and effective keyword search queries that will help find
    additional relevant information to answer the question comprehensively.

    Generate queries that:
    1. Break down complex questions into simpler, searchable components
    2. Use alternative phrasing or synonyms to increase chances of matches
    3. Focus on specific aspects of the question that need more context
    4. Are specific enough to find relevant information but not too narrow to miss important context
    """).strip()

SEARCH_QUERIES_USER_PROMPT = dedent("""
    Original Question: {original_query}

    Previous Search Queries:
    {formatted_prev_queries}

    Context Retrieved So Far:
    {formatted_context}

    Based on the above, generate 2-3 effective search queries that will help find information to answer the original question.
    This is attempt #{generation_attempt} at generating queries.
    """).strip()

# Prompts for grading context
GRADE_CONTEXT_SYSTEM_PROMPT = dedent("""
    You are an expert research assistant evaluating if the retrieved context is sufficient to answer a question.
    Your job is to determine if:
    1. The context is sufficient to provide a complete and accurate answer (FINISH)
    2. More context is needed to properly answer the question (CONTINUE)
    3. The retrieval strategy needs to be adjusted (RETRY_GENERATION)
    4. We should give up because the information is likely not in the documents (FAIL)

    Guidelines:
    - Choose FINISH if all aspects of the question can be answered with the current context
    - Choose CONTINUE if some parts of the question remain unanswered but more retrieval may help
    - Choose RETRY_GENERATION if the current retrieval strategy isn't working well
    - Choose FAIL only if we've made multiple attempts and still have no relevant information
    """).strip()

GRADE_CONTEXT_USER_PROMPT = dedent("""
    Original Question: {original_query}

    Current Iteration: {iterations} of {max_iterations}

    Retrieved Context:
    {formatted_context}

    Based on the above, determine if we should:
    - FINISH (context is sufficient to answer the question)
    - CONTINUE (need more context, continue retrieval)
    - RETRY_GENERATION (current strategy isn't working, try new queries)
    - FAIL (give up, information likely not in documents)
    """).strip()

# Prompts for generating final answer
FINAL_ANSWER_SYSTEM_PROMPT = dedent("""
    You are a meticulous research assistant tasked with answering questions based on specific context provided.
    Your response should:
    1. Be comprehensive, accurate, and directly address the question
    2. Be based ONLY on the provided context - do not include outside knowledge
    3. Cite specific sources from the context that support your answer using the format [X] where X is the source number
    4. If the context doesn't contain enough information to answer the question, say so clearly

    For your citations:
    - Include exact quotes from the source text
    - Specify the page number and filename for each quote
    - Link each citation to a specific claim in your answer

    Be thorough, accurate, and comprehensive in your response.
    """).strip()

FINAL_ANSWER_USER_PROMPT = dedent("""
    Question: {original_query}

    Context:
    {formatted_context}

    Based ONLY on the context provided above, answer the question. Include relevant citations for each key point in your answer.
    """).strip() 