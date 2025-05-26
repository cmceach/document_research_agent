import os
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from src.llm_calls.utils import deduplicate_search_results
from src.retriever.base_retriever import BaseRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaRetriever(BaseRetriever):
    """Chroma DB retriever for vector search."""
    
    def __init__(self, lazy_init=False):
        super().__init__()
        self.chroma_db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "document_chunks")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        
        # Initialize clients, but allow lazy initialization for testing
        self.chroma_client = None
        self.embedding_function = None
        self.collection = None
        
        if not lazy_init:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all clients - allows for delayed initialization in testing"""
        self.chroma_client = self._init_chroma_client()
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
            
            # Only log if there are documents (reduce noise for empty collections)
            doc_count = collection.count()
            if doc_count > 0:
                logger.info(f"Connected to collection '{self.collection_name}' with {doc_count} documents")
            return collection
            
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            raise
    
    def _normalize_filenames(self, filenames: List[str]) -> List[str]:
        """
        Normalize filenames by extracting just the basename if full paths are provided.
        
        Args:
            filenames: List of filenames or file paths
            
        Returns:
            List of normalized filenames (basenames only)
        """
        if not filenames:
            return []
            
        normalized = []
        for fname in filenames:
            # Extract just the basename if a path is provided
            basename = os.path.basename(fname)
            normalized.append(basename)
            if basename != fname:
                logger.debug(f"Normalized filename from '{fname}' to '{basename}'")
                
        return normalized
    
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
    
    def batch_retrieve_context(self, 
                             search_queries: List[str], 
                             filenames: List[str], 
                             top_k: int = 5,
                             batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Execute vector search in batches for better performance.
        
        Args:
            search_queries: List of search queries to use
            filenames: List of filenames to filter by
            top_k: Number of results to retrieve per query
            batch_size: Number of queries to process in each batch
            
        Returns:
            List of dicts with context information
        """
        results = []
        unique_contents = set()
        
        try:
            # Ensure client is initialized
            if self.collection is None:
                self._initialize_clients()
                
            # Normalize filenames to just basenames
            normalized_filenames = self._normalize_filenames(filenames)
            
            # Create a filter for filenames if provided
            where_filter = None
            if normalized_filenames:
                where_filter = {"filename": {"$in": normalized_filenames}}
                logger.debug(f"Filtering results by filenames: {normalized_filenames}")
            
            # Process queries in batches
            for i in range(0, len(search_queries), batch_size):
                batch_queries = search_queries[i:i + batch_size]
                logger.debug(f"Processing batch of {len(batch_queries)} queries")
                
                try:
                    # Execute vector search for the batch
                    search_results = self.collection.query(
                        query_texts=batch_queries,
                        n_results=top_k,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Process results for each query in the batch
                    if search_results and search_results.get("documents"):
                        for query_idx, documents in enumerate(search_results["documents"]):
                            if not documents:
                                continue
                                
                            # Extract results for this query
                            query_results = []
                            for doc_idx, document in enumerate(documents):
                                metadata = search_results["metadatas"][query_idx][doc_idx] if search_results.get("metadatas") else {}
                                
                                # Extract page number
                                page_number = 0
                                if "page_number" in metadata:
                                    try:
                                        page_number = int(metadata["page_number"])
                                    except (ValueError, TypeError):
                                        page_number = 0
                                
                                query_results.append({
                                    "text": document,
                                    "page": page_number,
                                    "filename": metadata.get("filename", "unknown")
                                })
                            
                            # Deduplicate results for this query
                            results.extend(deduplicate_search_results(query_results, unique_contents))
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(results)} unique context items across all batches")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch_retrieve_context: {e}")
            return []

    def retrieve_context(self, 
                        search_queries: List[str], 
                        filenames: List[str], 
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute vector search on Chroma DB with the provided queries.
        Uses batch processing for better performance with multiple queries.
        
        Args:
            search_queries: List of search queries to use
            filenames: List of filenames to filter by
            top_k: Number of results to retrieve per query
            
        Returns:
            List of dicts with context information
        """
        # Use batch processing if we have multiple queries
        if len(search_queries) > 1:
            return self.batch_retrieve_context(search_queries, filenames, top_k)
            
        # For single queries, use the original implementation
        results = []
        unique_contents = set()
        
        try:
            # Ensure client is initialized
            if self.collection is None:
                self._initialize_clients()
                
            # Normalize filenames to just basenames
            normalized_filenames = self._normalize_filenames(filenames)
            
            # Create a filter for filenames if provided
            where_filter = None
            if normalized_filenames:
                where_filter = {"filename": {"$in": normalized_filenames}}
                logger.debug(f"Filtering results by filenames: {normalized_filenames}")
            
            # Process each search query
            for query in search_queries:
                try:
                    logger.debug(f"Executing search for query: '{query}'")
                    
                    # Execute vector search
                    search_results = self.collection.query(
                        query_texts=[query],
                        n_results=top_k,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Process results
                    if search_results and search_results.get("documents") and search_results["documents"][0]:
                        docs_count = len(search_results["documents"][0])
                        logger.debug(f"Retrieved {docs_count} results for query: '{query}'")
                        
                        # Extract results into our consistent format
                        query_results = []
                        for i, document in enumerate(search_results["documents"][0]):
                            metadata = search_results["metadatas"][0][i] if search_results.get("metadatas") and search_results["metadatas"][0] else {}
                            
                            # Extract page number correctly, ensuring it's an integer
                            page_number = 0
                            if "page_number" in metadata:
                                try:
                                    page_number = int(metadata["page_number"])
                                except (ValueError, TypeError):
                                    page_number = 0
                            
                            # Get the filename from metadata
                            filename = metadata.get("filename", "unknown")
                            
                            query_results.append({
                                "text": document,
                                "page": page_number,
                                "filename": filename
                            })
                            
                        # Deduplicate results
                        results.extend(deduplicate_search_results(query_results, unique_contents))
                    else:
                        logger.debug(f"No results found for query: '{query}'")
                        
                except Exception as e:
                    logger.error(f"Error during search for query '{query}': {e}")
                    continue
            
            logger.debug(f"Retrieved {len(results)} unique context items across all queries")
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
                
            document_count = self.collection.count()
            
            # Get a sample of documents to examine metadata
            sample = self.collection.peek(limit=5)
            
            # Extract available filenames from metadata
            available_filenames = set()
            if sample and sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    if metadata and "filename" in metadata:
                        available_filenames.add(metadata["filename"])
            
            logger.debug(f"Sample filenames in collection: {list(available_filenames)}")
            
            return {
                "document_count": document_count,
                "sample_filenames": list(available_filenames)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"document_count": 0, "error": str(e)} 