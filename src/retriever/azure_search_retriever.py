import os
from typing import Dict, List, Any
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureSearchRetriever:
    """Azure AI Search retriever for hybrid search."""
    
    def __init__(self):
        self.search_endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")
        self.search_key = os.environ.get("AZURE_SEARCH_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        
        # Initialize clients
        self.search_client = self._init_search_client()
        self.openai_client = self._init_openai_client()
    
    def _init_search_client(self) -> SearchClient:
        """Initialize the Azure AI Search client."""
        try:
            if not all([self.search_endpoint, self.index_name, self.search_key]):
                raise ValueError("Azure Search configuration is incomplete")
            
            credential = AzureKeyCredential(self.search_key)
            return SearchClient(endpoint=self.search_endpoint, 
                              index_name=self.index_name, 
                              credential=credential)
        except Exception as e:
            logger.error(f"Failed to initialize Azure Search client: {e}")
            raise
    
    def _init_openai_client(self) -> OpenAI:
        """Initialize the OpenAI client."""
        try:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is missing")
            
            return OpenAI(api_key=self.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def retrieve_context(self, search_queries: List[str], filenames: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute hybrid search on Azure AI Search with the provided queries.
        
        Args:
            search_queries: List of search queries to use
            filenames: List of filenames to filter by
            top_k: Number of results to retrieve per query
            
        Returns:
            List of dicts with context information {"text": str, "page": int, "filename": str}
        """
        results = []
        unique_contents = set()  # Track unique content to avoid duplicates
        
        try:
            # First, construct the filter expression to only include the specified filenames
            if filenames:
                filter_expr = " or ".join([f"filename eq '{filename}'" for filename in filenames])
            else:
                filter_expr = None
                
            # Process each search query
            for query in search_queries:
                # Generate embedding for vector search component
                try:
                    query_vector = self.generate_embeddings(query)
                except Exception as e:
                    logger.warning(f"Could not generate embeddings for query '{query}': {e}")
                    continue
                
                # Execute hybrid search (combining keyword and vector search)
                try:
                    from azure.search.documents.models import VectorizedQuery
                    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="embedding")
                    
                    search_results = self.search_client.search(
                        search_text=query,  # Text query component
                        vector_queries=[vector_query],  # Vector query component
                        filter=filter_expr,
                        select=["content", "filename", "page_number"],
                        top=top_k
                    )
                    
                    # Process results
                    for result in search_results:
                        content = result["content"]
                        # Skip if we've already seen this content
                        if content in unique_contents:
                            continue
                            
                        unique_contents.add(content)
                        results.append({
                            "text": content,
                            "page": result.get("page_number", 0),
                            "filename": result.get("filename", "unknown")
                        })
                        
                except Exception as e:
                    logger.error(f"Error during search for query '{query}': {e}")
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {e}")
            return [] 