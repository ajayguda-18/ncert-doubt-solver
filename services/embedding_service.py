from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str, vector_db_path: str):
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        
        logger.info(f"Initializing embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Create vector DB directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB client (FIXED VERSION)
        logger.info(f"Initializing ChromaDB at: {vector_db_path}")
        try:
            self.client = chromadb.PersistentClient(
                path=vector_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("✅ ChromaDB client initialized")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def create_collection(self, collection_name: str):
        """Create or get collection"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Found existing collection '{collection_name}' with {self.collection.count()} documents")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error with collection: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks"""
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return chunks
    
    def store_chunks(self, chunks: List[Dict]):
        """Store chunks in vector database"""
        if not chunks:
            logger.warning("No chunks to store")
            return
        
        ids = [chunk["chunk_id"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        
        # Prepare metadata (exclude embedding and text)
        metadatas = []
        for chunk in chunks:
            metadata = {
                k: str(v) for k, v in chunk.items() 
                if k not in ["embedding", "text", "chunk_id"]
            }
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            try:
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                logger.info(f"Stored batch {i//batch_size + 1}: {batch_end - i} chunks")
            except Exception as e:
                logger.error(f"Error storing batch: {e}")
                raise
        
        total_docs = self.collection.count()
        logger.info(f"✅ Total documents in collection: {total_docs}")
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection.name,
                "count": count,
                "path": self.vector_db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None