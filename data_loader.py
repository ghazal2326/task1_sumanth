import os
import PyPDF2
from typing import List, Dict
from pathlib import Path

class PDFLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_pdfs(self) -> List[Dict[str, str]]:
        """Load all PDF files from data directory"""
        documents = []
        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}")
        
        for pdf_file in pdf_files:
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    
                    documents.append({
                        'file_name': pdf_file.name,
                        'content': text,
                        'source': str(pdf_file)
                    })
                    print(f"Loaded {pdf_file.name} with {len(pdf_reader.pages)} pages")
                    
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")
        
        return documents

    @staticmethod
    def validate_data_dir(data_dir: str):
        """Validate data directory exists and contains PDFs"""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")
        
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{data_dir}'")
        
        return True