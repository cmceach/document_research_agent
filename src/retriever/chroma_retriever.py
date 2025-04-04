import os
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaRetriever:
    """Chroma DB retriever for vector search."""
    
    def __init__(self, lazy_init=False):
        self.chroma_db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "document_chunks")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        
        # Initialize clients, but allow lazy initialization for testing
        self.chroma_client = None
        self.openai_client = None
        self.embedding_function = None
        self.collection = None
        
        if not lazy_init:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all clients - allows for delayed initialization in testing"""
        self.chroma_client = self._init_chroma_client()
        self.openai_client = self._init_openai_client()
        self.embedding_function = self._init_embedding_function()
        self.collection = self._get_or_create_collection()
    
    def _init_chroma_client(self) -> chromadb.PersistentClient:
        """Initialize the Chroma DB client."""
        try:
            # Configure settings with proper caching and timeout
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=False,
                persist_directory=self.chroma_db_path
            )
            
            return chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=settings
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma DB client: {e}")
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
    
    def _init_embedding_function(self):
        """Initialize the embedding function for Chroma."""
        try:
            # Use OpenAI embeddings
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name=self.embedding_model,
                dimensions=None  # Let the model determine dimensions
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create the collection."""
        try:
            # Try to get the existing collection
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"Successfully connected to collection '{self.collection_name}' with {collection.count()} documents")
            return collection
            
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            raise
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using OpenAI."""
        try:
            # Ensure client is initialized
            if self.openai_client is None:
                self._initialize_clients()
                
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def retrieve_context(self, 
                         search_queries: List[str], 
                         filenames: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute vector search on Chroma DB with the provided queries.
        
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
            # Ensure client is initialized
            if self.collection is None:
                self._initialize_clients()
                
            # Create a filter for filenames if provided
            where_filter = None
            if filenames:
                where_filter = {"filename": {"$in": filenames}}
                logger.info(f"Filtering results by filenames: {filenames}")
            
            # Process each search query
            for query in search_queries:
                try:
                    logger.info(f"Executing search for query: '{query}'")
                    
                    # Execute vector search
                    search_results = self.collection.query(
                        query_texts=[query],
                        n_results=top_k,
                        where=where_filter,
                        include=["documents", "metadatas"]
                    )
                    
                    # Process results
                    if search_results and search_results.get("documents") and search_results["documents"][0]:
                        logger.info(f"Retrieved {len(search_results['documents'][0])} results for query: '{query}'")
                        
                        # Extract results into our consistent format
                        for i, document in enumerate(search_results["documents"][0]):
                            # Skip if we've already seen this content
                            if document in unique_contents:
                                continue
                            
                            metadata = search_results["metadatas"][0][i] if search_results.get("metadatas") and search_results["metadatas"][0] else {}
                            
                            # Extract page number correctly, ensuring it's an integer
                            page_number = 0
                            if "page_number" in metadata:
                                try:
                                    page_number = int(metadata["page_number"])
                                except (ValueError, TypeError):
                                    page_number = 0
                            
                            unique_contents.add(document)
                            results.append({
                                "text": document,
                                "page": page_number,
                                "filename": metadata.get("filename", "unknown")
                            })
                    else:
                        logger.info(f"No results found for query: '{query}'")
                        
                except Exception as e:
                    logger.error(f"Error during search for query '{query}': {e}")
                    continue
            
            logger.info(f"Retrieved {len(results)} unique context items across all queries")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            # Ensure client is initialized
            if self.collection is None:
                self._initialize_clients()
                
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "embedding_model": self.embedding_model,
                "error": str(e)
            } 