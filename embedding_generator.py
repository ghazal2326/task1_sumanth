import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer"""
        print(f"Loading embedding model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling
                embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# Alternative using sentence-transformers (simpler)
try:
    from sentence_transformers import SentenceTransformer
    
    class SimpleEmbeddingGenerator:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model: {model_name}")
        
        def generate_embeddings(self, texts: List[str]) -> np.ndarray:
            return self.model.encode(texts, show_progress_bar=True)
            
except ImportError:
    print("sentence-transformers not available, using transformers instead")
    