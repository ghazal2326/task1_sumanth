import os
from config import Config
from data_loader import PDFLoader
from chunker import TextChunker
from embedding_generator import EmbeddingGenerator, SimpleEmbeddingGenerator
from vector_store import ChromaDBStore
from typing import List, Dict

class KnowledgeBaseBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.loader = PDFLoader(config.DATA_DIR)
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Use simple embedding generator if available, else fallback
        try:
            self.embedder = SimpleEmbeddingGenerator(config.EMBEDDING_MODEL)
        except:
            self.embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)
            
        self.vector_store = ChromaDBStore(
            persist_directory=config.VECTOR_DB_PATH,
            collection_name=config.COLLECTION_NAME
        )
    
    def build_knowledge_base(self):
        """Build the complete knowledge base pipeline"""
        print("Starting knowledge base construction...")
        
        # Step 1: Load PDFs
        print("\n1. Loading PDF documents...")
        documents = self.loader.load_pdfs()
        
        # Step 2: Chunk documents
        print("\n2. Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        print("\n3. Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(texts)
        
        # Step 4: Store in vector database
        print("\n4. Storing in vector database...")
        self.vector_store.store_documents(chunks, embeddings)
        
        print("\n✅ Knowledge base construction completed!")
        print(f"   - Documents processed: {len(documents)}")
        print(f"   - Total chunks created: {len(chunks)}")
        print(f"   - Vector database location: {self.config.VECTOR_DB_PATH}")
        print(f"   - Collection name: {self.config.COLLECTION_NAME}")
    
    def search(self, query: str, n_results: int = 5):
        """Search the knowledge base"""
        print(f"\nSearching for: '{query}'")
        results = self.vector_store.search(query, n_results)
        
        print(f"\nTop {n_results} results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
            print(f"Source: {metadata['file_name']}")
            print(f"Chunk: {metadata['chunk_index'] + 1}")
            print(f"Text: {doc[:200]}...")

def main():
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Build knowledge base
    kb_builder = KnowledgeBaseBuilder(config)
    
    try:
        # Build the knowledge base
        kb_builder.build_knowledge_base()
        
        # Example search
        print("\n" + "="*50)
        print("EXAMPLE SEARCH")
        print("="*50)
        kb_builder.search("What is the main topic of the documents?", n_results=3)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you have PDF files in the 'data' directory")

if __name__ == "__main__":
    main()
    