import chromadb
from chromadb.config import Settings
from typing import List, Dict
import numpy as np

class ChromaDBStore:
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base from PDF documents"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def store_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Store document chunks and their embeddings in ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
            ids.append(f"chunk_{i}_{chunk['metadata']['file_name']}")
        
        # Convert embeddings to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored {len(documents)} documents in vector database")
    
    def search(self, query: str, n_results: int = 5):
        """Search the vector database"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def get_collection_info(self):
        """Get information about the collection"""
        return self.collection.count()
    
    def reset_collection(self):
        """Reset the collection (delete all data)"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        print("Collection reset successfully")
        