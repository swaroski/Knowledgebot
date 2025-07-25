import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from .document_loader import DocumentChunk
from .gemini_client import gemini_client
from .config import config

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = 768  # Gemini embedding dimension
        self.index_path = config.FAISS_INDEX_PATH
        self.metadata_path = config.FAISS_INDEX_PATH.replace('.bin', '_metadata.pkl')
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.chunks = []
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to vector store"""
        try:
            embeddings = []
            valid_chunks = []
            
            for chunk in chunks:
                embedding = gemini_client.get_embeddings(chunk.text)
                if embedding:
                    # Normalize embedding for cosine similarity
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embedding_array = embedding_array / np.linalg.norm(embedding_array)
                    embeddings.append(embedding_array)
                    chunk.embedding = embedding_array
                    valid_chunks.append(chunk)
            
            if embeddings:
                embeddings_matrix = np.vstack(embeddings)
                self.index.add(embeddings_matrix)
                self.chunks.extend(valid_chunks)
                self.save_index()
                return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
        
        return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        try:
            if self.index.ntotal == 0:
                return []
            
            query_embedding = gemini_client.get_embeddings(query)
            if not query_embedding:
                return []
            
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            query_array = query_array / np.linalg.norm(query_array)
            
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Fallback keyword search"""
        query_words = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            if score > 0:
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents with metadata"""
        documents = {}
        for chunk in self.chunks:
            filename = chunk.metadata['filename']
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'file_type': chunk.metadata['file_type'],
                    'tags': chunk.metadata['tags'],
                    'total_chunks': chunk.metadata['total_chunks'],
                    'sections': []
                }
            
            if chunk.metadata['file_type'] == 'markdown' and chunk.metadata['section'] not in documents[filename]['sections']:
                documents[filename]['sections'].append(chunk.metadata['section'])
        
        return documents
    
    def save_index(self):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index and metadata"""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []

vector_store = VectorStore()