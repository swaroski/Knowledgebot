import re
from typing import List, Dict, Any
from pypdf import PdfReader
from .config import config

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.embedding = None

class DocumentLoader:
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
    
    def load_pdf(self, file_path: str, filename: str, tags: List[str] = None) -> List[DocumentChunk]:
        """Load and chunk PDF document"""
        chunks = []
        try:
            reader = PdfReader(file_path)
            full_text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            text_chunks = self._chunk_text(full_text)
            
            for i, chunk_text in enumerate(text_chunks):
                page_match = re.search(r'\[Page (\d+)\]', chunk_text)
                page_num = int(page_match.group(1)) if page_match else 1
                
                metadata = {
                    "filename": filename,
                    "file_type": "pdf",
                    "chunk_id": i,
                    "page_number": page_num,
                    "tags": tags or [],
                    "total_chunks": len(text_chunks)
                }
                chunks.append(DocumentChunk(chunk_text, metadata))
        
        except Exception as e:
            print(f"Error loading PDF {filename}: {e}")
        
        return chunks
    
    def load_markdown(self, file_path: str, filename: str, tags: List[str] = None) -> List[DocumentChunk]:
        """Load and chunk Markdown document"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            text_chunks = self._chunk_text(content)
            
            for i, chunk_text in enumerate(text_chunks):
                section = self._extract_section_header(chunk_text)
                
                metadata = {
                    "filename": filename,
                    "file_type": "markdown",
                    "chunk_id": i,
                    "section": section,
                    "tags": tags or [],
                    "total_chunks": len(text_chunks)
                }
                chunks.append(DocumentChunk(chunk_text, metadata))
        
        except Exception as e:
            print(f"Error loading Markdown {filename}: {e}")
        
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def _extract_section_header(self, text: str) -> str:
        """Extract section header from markdown chunk"""
        lines = text.split('\n')
        for line in lines:
            if line.startswith('#'):
                return line.strip()
        return "No section"

document_loader = DocumentLoader()