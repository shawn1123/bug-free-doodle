"""
Multithreaded Qdrant Vector Database Upload with LangChain

This script provides an optimized approach for uploading large datasets to Qdrant:
1. Multithreaded document processing 
2. Parallel embeddings generation
3. Concurrent uploads to Qdrant
"""

import os
import uuid
import datetime
import concurrent.futures
from typing import List, Dict, Any

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Multithreading and processing
import multiprocessing
import threading
from queue import Queue

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
COLLECTION_NAME = "my_documents"
EMBEDDING_DIMENSION = 1536  # For OpenAI embeddings

# Logging
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantUploader:
    def __init__(
        self, 
        qdrant_client: QdrantClient, 
        collection_name: str, 
        max_workers: int = None
    ):
        """
        Initialize Qdrant uploader with configurable threading
        
        :param qdrant_client: Qdrant client instance
        :param collection_name: Name of the collection to upload to
        :param max_workers: Number of threads/workers for parallel processing
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        
        # Use CPU count if not specified
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        
        # Embedding model
        self.embeddings = OpenAIEmbeddings()
        
        # Thread-safe upload tracking
        self.upload_lock = threading.Lock()
        self.total_uploaded = 0
        self.total_errors = 0

    def process_documents(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[dict]:
        """
        Load and process documents with parallel processing
        
        :param directory_path: Path to directory containing documents
        :param chunk_size: Size of text chunks
        :param chunk_overlap: Overlap between chunks
        :return: List of processed document chunks
        """
        # Load documents
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Prepare documents with thread-safe generation of metadata
        prepared_docs = []
        for chunk in chunks:
            doc_uuid = str(uuid.uuid4())
            
            # Extract filename from source
            source_file = chunk.metadata.get("source", "unknown")
            filename = os.path.basename(source_file)
            
            # Create document with metadata
            prepared_docs.append({
                "id": doc_uuid,
                "text": chunk.page_content,
                "metadata": {
                    "source": source_file,
                    "filename": filename,
                    "chunk_size": len(chunk.page_content),
                    "created_at": str(datetime.datetime.now()),
                }
            })
        
        return prepared_docs

    def generate_embeddings_batch(self, batch: List[dict]) -> List[dict]:
        """
        Generate embeddings for a batch of documents
        
        :param batch: List of documents to embed
        :return: List of documents with embeddings
        """
        # Extract texts
        texts = [doc["text"] for doc in batch]
        
        try:
            # Generate embeddings
            embedding_vectors = self.embeddings.embed_documents(texts)
            
            # Attach embeddings to documents
            for doc, embedding in zip(batch, embedding_vectors):
                doc["embedding"] = embedding
            
            return batch
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def upload_batch(self, batch: List[dict]) -> None:
        """
        Upload a batch of points to Qdrant
        
        :param batch: Batch of documents with embeddings
        """
        try:
            # Prepare points for Qdrant
            points = [
                models.PointStruct(
                    id=doc["id"],
                    vector=doc["embedding"],
                    payload={
                        "text": doc["text"],
                        **doc["metadata"]
                    }
                ) for doc in batch
            ]
            
            # Thread-safe upload
            with self.upload_lock:
                # Upload batch to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                # Update tracking
                self.total_uploaded += len(points)
                logger.info(f"Uploaded batch of {len(points)} documents")
        
        except Exception as e:
            with self.upload_lock:
                self.total_errors += len(batch)
                logger.error(f"Error uploading batch: {e}")

    def process_and_upload(self, documents: List[dict], batch_size: int = 100) -> None:
        """
        Process and upload documents using parallel processing
        
        :param documents: List of documents to process and upload
        :param batch_size: Size of batches for processing
        """
        # Split documents into batches for processing
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        # Parallel processing of batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Stage 1: Generate Embeddings
            embedding_futures = [
                executor.submit(self.generate_embeddings_batch, batch) 
                for batch in batches
            ]
            
            # Stage 2: Upload Batches
            upload_futures = []
            for future in concurrent.futures.as_completed(embedding_futures):
                embedded_batch = future.result()
                if embedded_batch:
                    upload_futures.append(
                        executor.submit(self.upload_batch, embedded_batch)
                    )
            
            # Wait for all uploads to complete
            concurrent.futures.wait(upload_futures)

def get_or_create_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create Qdrant collection if it doesn't exist
    
    :param client: Qdrant client
    :param collection_name: Name of the collection
    """
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            )
        )
        logger.info(f"Collection '{collection_name}' created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

def main():
    # Connect to Qdrant
    # Local Qdrant instance
    client = QdrantClient(host="localhost", port=6333)
    
    # Qdrant Cloud example (uncomment and configure)
    # client = QdrantClient(
    #     url="https://your-cluster-url.qdrant.io",
    #     api_key="your-api-key"
    # )
    
    # Create collection if it doesn't exist
    get_or_create_collection(client, COLLECTION_NAME)
    
    # Initialize uploader
    uploader = QdrantUploader(
        qdrant_client=client, 
        collection_name=COLLECTION_NAME,
        max_workers=None  # Uses 2 * CPU count by default
    )
    
    # Process documents from a directory
    documents = uploader.process_documents("./data")
    
    # Upload documents with multithreading
    uploader.process_and_upload(documents, batch_size=50)
    
    # Final logging
    logger.info(f"Upload complete. Total documents: {uploader.total_uploaded}")
    if uploader.total_errors > 0:
        logger.warning(f"Encountered {uploader.total_errors} errors during upload")

if __name__ == "__main__":
    main()

# Example search function (similar to previous implementation)
def search_example(client: QdrantClient, collection_name: str, query: str, top_k: int = 5):
    """
    Perform a similarity search on the Qdrant collection
    
    :param client: Qdrant client
    :param collection_name: Name of the collection to search
    :param query: Search query string
    :param top_k: Number of top results to return
    """
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings()
    
    # Initialize the Qdrant retriever
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    # Search the database
    docs = qdrant.similarity_search(query, k=top_k)
    
    # Print results
    for i, doc in enumerate(docs):
        print(f"Result {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)

# Performance monitoring decorator (optional)
def performance_monitor(func):
    """
    Decorator to monitor function performance
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper
