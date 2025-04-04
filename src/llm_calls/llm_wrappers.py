import os
import json
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMWrappers:
    """Wrappers for LLM calls in the Document Research Agent."""
    
    def __init__(self, lazy_init=False):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.chat_model = os.environ.get("OPENAI_CHAT_MODEL_NAME", "o3-mini")
        
        # Initialize OpenAI client
        self.client = None
        if not lazy_init:
            self.client = self._init_openai_client()
    
    def _init_openai_client(self) -> OpenAI:
        """Initialize the OpenAI client."""
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key is missing")
            
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _ensure_client_initialized(self):
        """Ensure the OpenAI client is initialized before use"""
        if self.client is None:
            self.client = self._init_openai_client()
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_search_queries_llm(self, 
                                   original_query: str, 
                                   retrieved_context: List[Dict[str, Any]],
                                   previous_queries: List[str],
                                   generation_attempt: int = 1) -> List[str]:
        """
        Generate search queries based on the original query and retrieved context.
        
        Args:
            original_query: The user's original question
            retrieved_context: Previously retrieved context chunks
            previous_queries: Previously generated search queries
            generation_attempt: Current attempt number for generating queries
            
        Returns:
            List of search query strings
        """
        try:
            self._ensure_client_initialized()
            
            # Prepare context for the prompt
            formatted_context = ""
            if retrieved_context:
                for i, item in enumerate(retrieved_context):
                    formatted_context += f"[{i+1}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
            
            formatted_prev_queries = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
            
            # Create the prompt
            system_prompt = """You are a skilled researcher with the ability to formulate targeted search queries to find information in a document.
Based on the original question and any previously retrieved context, generate specific and effective keyword search queries that will help find
additional relevant information to answer the question comprehensively.

Generate queries that:
1. Break down complex questions into simpler, searchable components
2. Use alternative phrasing or synonyms to increase chances of matches
3. Focus on specific aspects of the question that need more context
4. Are specific enough to find relevant information but not too narrow to miss important context

Return ONLY a JSON array of search queries, with no additional text or explanation."""
            
            user_prompt = f"""Original Question: {original_query}

Previous Search Queries:
{formatted_prev_queries}

Context Retrieved So Far:
{formatted_context or "No context retrieved yet."}

Based on the above, generate 2-3 effective search queries that will help find information to answer the original question.
This is attempt #{generation_attempt} at generating queries.

Return ONLY a JSON array of search queries. For example: ["query 1", "query 2", "query 3"]"""
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            content = response.choices[0].message.content
            try:
                # Try to parse as JSON
                queries = json.loads(content)
                if not isinstance(queries, list):
                    logger.warning(f"LLM did not return a list: {content}")
                    queries = [content]
            except json.JSONDecodeError:
                # If not valid JSON, use the raw content as a single query
                logger.warning(f"Failed to parse LLM response as JSON: {content}")
                # Try to extract array-like content from the text
                import re
                match = re.search(r'\[(.*)\]', content)
                if match:
                    try:
                        queries = json.loads(f"[{match.group(1)}]")
                    except:
                        queries = [content]
                else:
                    queries = [content]
            
            # Filter out any empty queries and limit to reasonable length
            queries = [q.strip() for q in queries if q and q.strip()]
            queries = [q[:100] for q in queries]  # Limit length of each query
            
            return queries[:3]  # Return at most 3 queries
            
        except Exception as e:
            logger.error(f"Error in query generation: {e}")
            # Fall back to a simple query based on the original question
            return [original_query]
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def grade_context_llm(self, 
                         original_query: str, 
                         retrieved_context: List[Dict[str, Any]],
                         iterations: int,
                         max_iterations: int) -> str:
        """
        Evaluate if the retrieved context is sufficient to answer the query.
        
        Args:
            original_query: The user's original query
            retrieved_context: List of context items retrieved so far
            iterations: Current iteration count
            max_iterations: Maximum allowed iterations
            
        Returns:
            Decision string: "FINISH", "CONTINUE", "RETRY_GENERATION", or "FAIL"
        """
        try:
            self._ensure_client_initialized()
            
            # Prepare context for the prompt
            formatted_context = ""
            if retrieved_context:
                for i, item in enumerate(retrieved_context):
                    formatted_context += f"[{i+1}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
            
            # Create the prompt
            system_prompt = """You are an expert researcher who evaluates if the information retrieved from documents is sufficient to answer a question.
Your task is to determine if the retrieved context contains enough information to provide a complete and accurate answer to the original question.

Consider the following:
1. Is there enough context to fully answer all aspects of the question?
2. Are there any missing pieces of information that would make the answer incomplete?
3. Is the information relevant and directly addresses the question?
4. Is the information from reliable sources within the document set?

Return ONLY one of these decisions with no additional explanation:
- "FINISH" if the context is sufficient to provide a complete answer
- "CONTINUE" if more information is needed but the current context is relevant
- "RETRY_GENERATION" if the current search approach isn't yielding relevant information and we should try different queries
- "FAIL" if after multiple attempts, it's clear the documents don't contain information to answer the question"""
            
            user_prompt = f"""Original Question: {original_query}

Retrieved Context:
{formatted_context or "No context retrieved yet."}

Current iteration: {iterations} out of maximum {max_iterations} iterations.

Based on the context above, should I:
1. FINISH - The context contains sufficient information to answer the question
2. CONTINUE - Need more information, continue searching with similar queries
3. RETRY_GENERATION - Current search approach isn't working, try different queries
4. FAIL - The documents likely don't contain the information needed

Return ONLY one decision word: FINISH, CONTINUE, RETRY_GENERATION, or FAIL."""
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=20
            )
            
            content = response.choices[0].message.content.strip().upper()
            
            # Extract the decision - look for one of the valid decisions
            valid_decisions = ["FINISH", "CONTINUE", "RETRY_GENERATION", "FAIL"]
            for decision in valid_decisions:
                if decision in content:
                    return decision
            
            # If no valid decision found, determine based on context and iterations
            if not retrieved_context:
                return "CONTINUE" if iterations < max_iterations - 1 else "FAIL"
            elif iterations >= max_iterations - 1:
                return "FINISH"  # Last iteration, try to finish with what we have
            else:
                return "CONTINUE"  # Default to continue
            
        except Exception as e:
            logger.error(f"Error in context grading: {e}")
            # Default behavior based on iterations
            if iterations >= max_iterations - 1:
                return "FINISH"
            return "CONTINUE"
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_final_answer_llm(self, 
                                 original_query: str, 
                                 retrieved_context: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate the final answer based on retrieved context.
        
        Args:
            original_query: The user's original query
            retrieved_context: List of retrieved context items
            
        Returns:
            Tuple of (answer_string, citations_list)
        """
        try:
            self._ensure_client_initialized()
            
            if not retrieved_context:
                return "Information not found in provided documents", []
            
            # Prepare context for the prompt
            formatted_context = ""
            for i, item in enumerate(retrieved_context):
                formatted_context += f"[{i+1}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
            
            # Create the prompt
            system_prompt = """You are an AI research assistant that provides accurate, factual answers based ONLY on the provided document context.
Your task is to synthesize information from the context to answer the user's question completely and accurately.

Important rules:
1. ONLY use information found in the provided context. DO NOT use external knowledge or make up information.
2. If the context doesn't contain enough information to answer the question, state "Information not found in provided documents."
3. Cite your sources for each piece of information using the reference numbers from the context.
4. Be concise but thorough in your answer.
5. For each piece of information you use, you must provide a citation.

Return your response as a JSON object with two fields:
1. "answer": A string containing your synthesized answer with citation references like [1], [2], etc.
2. "citations": An array of objects, each with "text", "page", and "filename" fields corresponding to the citations used."""
            
            user_prompt = f"""Original Question: {original_query}

Context:
{formatted_context}

Based ONLY on the context above, answer the question. Include citation references like [1], [2] in your answer 
to indicate which parts of the context you're using.

Return your answer in this JSON format:
{{
  "answer": "Your synthesized answer with citation references [1], [2], etc.",
  "citations": [
    {{"text": "Exact text quoted from context item 1", "page": 5, "filename": "document1.pdf"}},
    {{"text": "Exact text quoted from context item 2", "page": 10, "filename": "document2.pdf"}}
  ]
}}

If you cannot answer the question from the provided context, return:
{{
  "answer": "Information not found in provided documents",
  "citations": []
}}"""
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            try:
                # Parse the JSON response
                result = json.loads(content)
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                
                # Validate answer
                if not answer or answer.strip() == "":
                    answer = "Information not found in provided documents"
                    citations = []
                
                # If answer indicates information not found, ensure citations is empty
                if "information not found" in answer.lower():
                    citations = []
                
                # Validate each citation has required fields
                valid_citations = []
                for citation in citations:
                    if isinstance(citation, dict) and "text" in citation:
                        # Ensure all required fields exist
                        valid_citation = {
                            "text": citation.get("text", ""),
                            "page": citation.get("page", 0),
                            "filename": citation.get("filename", "unknown")
                        }
                        valid_citations.append(valid_citation)
                
                return answer, valid_citations
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {content}")
                # Extract answer from non-JSON response
                if "information not found" in content.lower():
                    return "Information not found in provided documents", []
                
                # Try to create a reasonable answer from the raw text
                return content, [{"text": "Citation information could not be parsed", "page": 0, "filename": "unknown"}]
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "An error occurred while generating the answer", [] 