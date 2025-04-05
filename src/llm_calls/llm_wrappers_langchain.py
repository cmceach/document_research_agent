import os
from typing import Dict, List, Any, Tuple, Optional, Literal
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class Citation(BaseModel):
    """Citation for a piece of information in a document."""
    text: str = Field(..., description="The exact text quoted from the source")
    page: int = Field(..., description="The page number where the text appears")
    filename: str = Field(..., description="The filename of the source document")

class FinalAnswer(BaseModel):
    """Final answer with citations."""
    answer: str = Field(..., description="The synthesized answer to the user's question")
    citations: List[Citation] = Field(description="List of citations supporting the answer")

class ContextDecision(BaseModel):
    """Decision about the quality of retrieved context."""
    decision: Literal["FINISH", "CONTINUE", "RETRY_GENERATION", "FAIL"] = Field(
        ..., 
        description="Decision on whether the context is sufficient"
    )

class SearchQueries(BaseModel):
    """Search queries for retrieving context."""
    queries: List[str] = Field(
        ..., 
        description="List of search queries to find relevant information"
    )

class LLMWrappers:
    """Wrappers for LLM calls in the Document Research Agent using LangChain."""
    
    def __init__(self, lazy_init=False):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = os.environ.get("OPENAI_CHAT_MODEL_NAME", "o3-mini")
        
        # Initialize models unless lazy initialization is requested
        self.chat_model = None
        
        # Load models immediately unless lazy_init is True
        if not lazy_init:
            self._ensure_models_initialized()

    def _ensure_models_initialized(self):
        """
        Initialize the ChatOpenAI model if it hasn't been already.
        """
        if not self.chat_model:
            from langchain_openai import ChatOpenAI
            
            # Initialize the model - o3-mini doesn't support temperature parameter
            if self.model_name == "o3-mini":
                self.chat_model = ChatOpenAI(
                    model_name=self.model_name
                )
            else:
                self.chat_model = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0.0  # Lower temperature for more consistent outputs
                )
            
            logging.info(f"LLM models initialized using {self.model_name}")
        
        return self.chat_model
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_search_queries_llm(self, 
                                   original_query: str, 
                                   retrieved_context: List[Dict[str, Any]],
                                   previous_queries: List[str],
                                   generation_attempt: int = 1) -> List[str]:
        """
        Generate search queries based on the original query and retrieved context.
        """
        try:
            self._ensure_models_initialized()
            
            # Prepare context for the prompt
            formatted_context = ""
            if retrieved_context:
                for i, item in enumerate(retrieved_context):
                    formatted_context += f"[{i+1}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
            
            formatted_prev_queries = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
            
            # Create structured output chain with JSON mode instead of function_calling
            structured_llm = self.chat_model.with_structured_output(
                SearchQueries
            )
            
            system_prompt = """You are a skilled researcher with the ability to formulate targeted search queries to find information in a document.
Based on the original question and any previously retrieved context, generate specific and effective keyword search queries that will help find
additional relevant information to answer the question comprehensively.

Generate queries that:
1. Break down complex questions into simpler, searchable components
2. Use alternative phrasing or synonyms to increase chances of matches
3. Focus on specific aspects of the question that need more context
4. Are specific enough to find relevant information but not too narrow to miss important context"""
            
            user_prompt = f"""Original Question: {original_query}

Previous Search Queries:
{formatted_prev_queries}

Context Retrieved So Far:
{formatted_context or "No context retrieved yet."}

Based on the above, generate 2-3 effective search queries that will help find information to answer the original question.
This is attempt #{generation_attempt} at generating queries."""
            
            # Use messages list format
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Result is a Pydantic model with a queries attribute
            queries = result.queries
            
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
        """
        try:
            self._ensure_models_initialized()
            
            # Prepare context for the prompt
            formatted_context = ""
            if retrieved_context:
                for i, item in enumerate(retrieved_context):
                    formatted_context += f"[{i+1}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
            
            # Create structured output chain with JSON mode instead of function_calling
            structured_llm = self.chat_model.with_structured_output(
                ContextDecision
            )
            
            system_prompt = """You are an expert researcher who evaluates if the information retrieved from documents is sufficient to answer a question.

Your task is to determine if the retrieved context contains enough information to provide a complete and accurate answer to the original question.

Consider the following:
1. Is there enough context to fully answer all aspects of the question?
2. Are there any missing pieces of information that would make the answer incomplete?
3. Is the information relevant and directly addresses the question?
4. Is the information from reliable sources within the document set?

Return one of these decisions:
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
4. FAIL - The documents likely don't contain the information needed"""
            
            # Use messages list format
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            return result.decision
            
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
        """
        try:
            self._ensure_models_initialized()
            
            if not retrieved_context:
                return "Information not found in provided documents", []
            
            # Prepare context for the prompt
            formatted_context = ""
            context_items = {}  # Store context items by their index for easier citation
            
            for i, item in enumerate(retrieved_context):
                # Use 1-indexed numbering in the prompt
                index = i + 1
                formatted_context += f"[{index}] Text: {item['text']}\nSource: {item['filename']}, Page: {item['page']}\n\n"
                # Store the context item with its index
                context_items[index] = item
            
            # Create structured output chain with JSON mode instead of function_calling
            structured_llm = self.chat_model.with_structured_output(
                FinalAnswer
            )
            
            system_prompt = """You are an AI research assistant that provides accurate, factual answers based ONLY on the provided document context.
Your task is to synthesize information from the context to answer the user's question completely and accurately.

When providing an answer:
1. Only include information that is present in the context
2. Do not make up or infer information that is not explicitly stated
3. Cite specific sources from the context that support your answer using the format [X] where X is the source number
4. If the context doesn't contain enough information to answer the question, say so clearly

For your citations:
- Include exact quotes from the source text
- Specify the page number and filename for each quote
- Link each citation to a specific claim in your answer

Be thorough, accurate, and comprehensive in your response."""
            
            user_prompt = f"""Question: {original_query}

Context:
{formatted_context}

Based ONLY on the context provided above, answer the question. Include relevant citations for each key point in your answer."""
            
            # Use messages list format
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Process the citations to ensure they match the format expected by the agent
            processed_citations = []
            for citation in result.citations:
                processed_citations.append({
                    "text": citation.text,
                    "page": citation.page,
                    "filename": citation.filename
                })
            
            return result.answer, processed_citations
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "An error occurred while generating the answer", [] 