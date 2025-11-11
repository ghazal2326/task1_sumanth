from typing import List, Dict
import re

class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks with metadata"""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_text(doc['content'])
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = {
                    'text': chunk_text,
                    'source': doc['file_name'],
                    'chunk_id': i,
                    'total_chunks': len(doc_chunks),
                    'metadata': {
                        'file_name': doc['file_name'],
                        'source': doc['source'],
                        'chunk_index': i
                    }
                }
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start += self.chunk_size - self.chunk_overlap
            
            # Ensure we don't get stuck in infinite loop
            if start >= len(words):
                break
        
        return chunks

class SentenceChunker(TextChunker):
    """Alternative chunker that respects sentence boundaries"""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence boundaries"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks