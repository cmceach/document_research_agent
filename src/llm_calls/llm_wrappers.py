import os
from typing import Dict, List, Any, Tuple, Optional, Literal, Union
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.callbacks.manager import get_openai_callback

# Import utility functions
from src.llm_calls.utils import format_context, format_previous_queries, clean_query_results, optimize_context_for_prompt
from src.llm_calls.prompts import (
    SEARCH_QUERIES_SYSTEM_PROMPT,
    SEARCH_QUERIES_USER_PROMPT,
    GRADE_CONTEXT_SYSTEM_PROMPT,
    GRADE_CONTEXT_USER_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_PROMPT
)

# Setup logging
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

class TokenUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = Field(0, description="Number of tokens used in prompts")
    completion_tokens: int = Field(0, description="Number of tokens in completions")
    total_tokens: int = Field(0, description="Total tokens used")

class LLMWrappers:
    """Wrapper for LLM calls in the Document Research Agent using LangChain."""
    
    def __init__(self, lazy_init: bool = False):
        """
        Initialize the LLM wrapper with LangChain models.
        
        Args:
            lazy_init: If True, delay model initialization until first use
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = os.environ.get("OPENAI_CHAT_MODEL_NAME", "gpt-4o")
        
        # Initialize models unless lazy initialization is requested
        self.chat_model = None
        
        # Initialize token usage counters
        self.token_usage = TokenUsage()
        
        # Load models immediately unless lazy_init is True
        if not lazy_init:
            self._ensure_models_initialized()
    
    def _ensure_models_initialized(self) -> ChatOpenAI:
        """
        Initialize the ChatOpenAI model if it hasn't been already.
        
        Returns:
            The initialized ChatOpenAI model
        """
        if not self.chat_model:
            # Check if the model is o3-mini which doesn't support temperature
            if "o3-mini" in self.model_name:
                self.chat_model = ChatOpenAI(
                    model_name=self.model_name
                )
                logger.info(f"LLM models initialized using {self.model_name} without temperature (not supported)")
            else:
                # Always use temperature=0 for deterministic results for models that support it
                self.chat_model = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0.0
                )
                logger.info(f"LLM models initialized using {self.model_name} with temperature=0.0")
        
        return self.chat_model
    
    def _get_structured_llm(self, model_class: type) -> Any:
        """
        Get a structured LLM with the specified output schema.
        
        Args:
            model_class: The Pydantic model class for structured output
            
        Returns:
            A structured LLM with the specified schema
        """
        self._ensure_models_initialized()
        return self.chat_model.with_structured_output(model_class, method="function_calling")
    
    def _create_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """
        Create a messages list for LLM invocation.
        
        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The user prompt for the LLM
            
        Returns:
            A list of message dictionaries
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _update_token_usage(self, usage: Dict[str, int]) -> None:
        """
        Update token usage statistics
        
        Args:
            usage: Token usage dictionary from the OpenAI callback
        """
        self.token_usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.token_usage.completion_tokens += usage.get("completion_tokens", 0)
        self.token_usage.total_tokens += usage.get("total_tokens", 0)
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get the current token usage statistics
        
        Returns:
            Dictionary with token usage information
        """
        return {
            "prompt_tokens": self.token_usage.prompt_tokens,
            "completion_tokens": self.token_usage.completion_tokens,
            "total_tokens": self.token_usage.total_tokens
        }
    
    def reset_token_usage(self) -> None:
        """Reset the token usage counters to zero"""
        self.token_usage = TokenUsage()
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_search_queries_llm(
        self, 
        original_query: str, 
        retrieved_context: List[Dict[str, Any]],
        previous_queries: List[str],
        generation_attempt: int = 1
    ) -> List[str]:
        """
        Generate search queries based on the original query and retrieved context.
        
        Args:
            original_query: The user's original query
            retrieved_context: Previously retrieved context chunks
            previous_queries: Previously generated search queries
            generation_attempt: Current attempt number for generating queries
            
        Returns:
            List of search query strings
        """
        try:
            # Create structured LLM
            structured_llm = self._get_structured_llm(SearchQueries)
            
            # Format context and previous queries with token optimization
            formatted_context = format_context(retrieved_context, max_items=8, max_chars=200)
            formatted_prev_queries = format_previous_queries(previous_queries)
            
            # Create messages using prompts from prompts.py
            user_prompt = SEARCH_QUERIES_USER_PROMPT.format(
                original_query=original_query,
                formatted_prev_queries=formatted_prev_queries,
                formatted_context=formatted_context,
                generation_attempt=generation_attempt
            )
            
            messages = self._create_messages(SEARCH_QUERIES_SYSTEM_PROMPT, user_prompt)
            
            # Track token usage
            with get_openai_callback() as cb:
                result = structured_llm.invoke(messages)
                # Update token usage from callback
                self._update_token_usage({
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens
                })
                logger.debug(f"Search query generation used {cb.total_tokens} tokens (prompt: {cb.prompt_tokens}, completion: {cb.completion_tokens})")
            
            # Clean and return the query results
            return clean_query_results(result.queries)
            
        except Exception as e:
            logger.error(f"Error in query generation: {e}")
            # Fall back to a simple query based on the original question
            return [original_query]
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def grade_context_llm(
        self, 
        original_query: str, 
        retrieved_context: List[Dict[str, Any]],
        iterations: int,
        max_iterations: int
    ) -> str:
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
            # Create structured LLM
            structured_llm = self._get_structured_llm(ContextDecision)
            
            # Format the context for the prompt with token optimization
            formatted_context = format_context(retrieved_context, max_items=10, max_chars=250)
            
            # Create messages using prompts from prompts.py
            user_prompt = GRADE_CONTEXT_USER_PROMPT.format(
                original_query=original_query,
                iterations=iterations,
                max_iterations=max_iterations,
                formatted_context=formatted_context
            )
            
            messages = self._create_messages(GRADE_CONTEXT_SYSTEM_PROMPT, user_prompt)
            
            # Track token usage
            with get_openai_callback() as cb:
                result = structured_llm.invoke(messages)
                self._update_token_usage({
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens
                })
                logger.debug(f"Context grading used {cb.total_tokens} tokens")
            
            return result.decision
            
        except Exception as e:
            logger.error(f"Error in context grading: {e}")
            
            # If we hit maximum iterations, finish anyway
            if iterations >= max_iterations:
                return "FINISH"
            
            # Otherwise continue to be safe
            return "CONTINUE"
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_final_answer_llm(
        self, 
        original_query: str, 
        retrieved_context: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a final answer with citations based on the retrieved context.
        
        Args:
            original_query: The user's original query
            retrieved_context: List of context items retrieved
            
        Returns:
            Tuple containing the answer string and a list of citation dictionaries
        """
        # If no context is provided, return early
        if not retrieved_context:
            return "Information not found in provided documents", []
            
        try:
            # Create structured LLM
            structured_llm = self._get_structured_llm(FinalAnswer)
            
            # Optimize context to fit within token limits (target ~6000 tokens for context)
            optimized_context = optimize_context_for_prompt(retrieved_context, target_tokens=6000)
            if len(optimized_context) < len(retrieved_context):
                logger.warning(f"Context was optimized from {len(retrieved_context)} to {len(optimized_context)} items to fit within token limits")
            
            # Format for the prompt with optimized parameters
            formatted_context = format_context(optimized_context, max_items=15, max_chars=400)
            
            # Create messages using prompts from prompts.py
            user_prompt = FINAL_ANSWER_USER_PROMPT.format(
                original_query=original_query,
                formatted_context=formatted_context
            )
            
            messages = self._create_messages(FINAL_ANSWER_SYSTEM_PROMPT, user_prompt)
            
            # Track token usage
            with get_openai_callback() as cb:
                result = structured_llm.invoke(messages)
                self._update_token_usage({
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens
                })
                logger.debug(f"Final answer generation used {cb.total_tokens} tokens")
            
            # If the answer indicates no information was found, return empty citations
            if result.answer == "Information not found in provided documents":
                return result.answer, []
            
            # Process the citations to ensure they match the format expected by the agent
            processed_citations = [
                {
                    "text": citation.text,
                    "page": citation.page,
                    "filename": citation.filename
                }
                for citation in result.citations
            ]
            
            return result.answer, processed_citations
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "An error occurred while generating the answer", [] 