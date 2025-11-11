import os
from dataclasses import dataclass

@dataclass
class Config:
    # Data paths
    DATA_DIR = "data"
    VECTOR_DB_PATH = "chroma_db"
    
    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector database
    COLLECTION_NAME = "knowledge_base"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)