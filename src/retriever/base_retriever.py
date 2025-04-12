from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from src.utils.cache_utils import cached

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """Base class for document retrievers."""
    
    def __init__(self):
        self.openai_api_key = None
        self.embedding_model = None
        self.openai_client = None
        
    def _init_openai_client(self) -> OpenAI:
        """Initialize the OpenAI client."""
        try:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is missing")
            
            return OpenAI(api_key=self.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    @cached(ttl=86400)  # Cache embeddings for 24 hours
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using OpenAI."""
        try:
            if self.openai_client is None:
                self.openai_client = self._init_openai_client()
                
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @abstractmethod
    def retrieve_context(self, 
                        search_queries: List[str], 
                        filenames: List[str], 
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute search with the provided queries.
        
        Args:
            search_queries: List of search queries to use
            filenames: List of filenames to filter by
            top_k: Number of results to retrieve per query
            
        Returns:
            List of dicts with context information {"text": str, "page": int, "filename": str}
        """
        pass 