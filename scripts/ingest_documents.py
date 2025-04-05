#!/usr/bin/env python3
"""
Script to ingest sample documents into ChromaDB for testing the Document Research Agent.
"""

import os
import logging
import argparse
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only include non-empty pages
                pages.append({
                    "text": text,
                    "page_number": i + 1,
                    "filename": os.path.basename(pdf_path)
                })
        
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return []

def ingest_documents(document_paths, chunk_size=1000, chunk_overlap=200):
    """Ingest documents into ChromaDB"""
    # Get API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key is missing. Please set OPENAI_API_KEY in .env file")
        return
    
    # Initialize embedding function
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002"
    )
    
    # Initialize Chroma client using the new API
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    # Create or get collection
    collection_name = "document_chunks"
    
    # Check if collection exists and delete it to start fresh
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"No existing collection to delete: {collection_name}")
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    logger.info(f"Created collection: {collection_name}")
    
    # Process each document
    document_count = 0
    chunk_count = 0
    
    for doc_path in document_paths:
        if not os.path.exists(doc_path):
            logger.warning(f"Document not found: {doc_path}")
            continue
            
        # Extract text from document
        logger.info(f"Extracting text from {doc_path}")
        pages = extract_text_from_pdf(doc_path)
        
        # Get just the filename (not the full path)
        filename = os.path.basename(doc_path)
        
        # Add each page as a separate document
        for page in pages:
            try:
                # Create a unique ID
                doc_id = f"{filename}_page_{page['page_number']}"
                
                # Add to collection
                collection.add(
                    ids=[doc_id],
                    documents=[page["text"]],
                    metadatas=[{
                        "filename": filename,  # Store just the filename, not the full path
                        "page_number": page["page_number"]
                    }]
                )
                chunk_count += 1
            except Exception as e:
                logger.error(f"Error adding document chunk: {e}")
        
        document_count += 1
        logger.info(f"Processed document: {doc_path}")
    
    logger.info(f"Ingestion complete. Processed {document_count} documents with {chunk_count} chunks.")
    
    # Verify the collection
    collection_stats = collection.count()
    logger.info(f"Collection '{collection_name}' now contains {collection_stats} documents.")
    
    return collection_stats

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--directory", type=str, default="test_data", 
                        help="Directory containing PDF documents to ingest")
    args = parser.parse_args()
    
    # Get all PDF files in the directory
    document_paths = []
    for filename in os.listdir(args.directory):
        if filename.endswith(".pdf"):
            document_paths.append(os.path.join(args.directory, filename))
    
    if not document_paths:
        logger.error(f"No PDF documents found in {args.directory}")
        return
    
    logger.info(f"Found {len(document_paths)} PDF documents to ingest")
    
    # Ingest documents
    ingest_documents(document_paths)

if __name__ == "__main__":
    main() 