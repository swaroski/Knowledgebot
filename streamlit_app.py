import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import tempfile
import os
import uuid
import re
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Q&A Knowledge Base",
    page_icon="🧠",
    layout="wide"
)

# Configuration
class Config:
    def __init__(self):
        # Try Streamlit secrets first, then environment variables
        try:
            self.GEMINI_API_KEY = st.secrets["general"]["GEMINI_API_KEY"]
            self.FAISS_INDEX_PATH = st.secrets["general"].get("FAISS_INDEX_PATH", "faiss_index.bin")
            self.CHUNK_SIZE = int(st.secrets["general"].get("CHUNK_SIZE", 500))
            self.CHUNK_OVERLAP = int(st.secrets["general"].get("CHUNK_OVERLAP", 50))
        except:
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            self.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
            self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
            self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

config = Config()

# Document chunk class
class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.embedding = None

# Initialize Gemini
@st.cache_resource
def init_gemini():
    if not config.GEMINI_API_KEY:
        st.error("Please set GEMINI_API_KEY in secrets.toml or environment variables")
        st.stop()
    
    genai.configure(api_key=config.GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-pro")

# Document processing functions
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or not text.strip():
        return []
    
    words = text.split()
    if not words:
        return []
    
    # If text is very short, return as single chunk
    if len(words) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    overlap = max(0, min(chunk_overlap, chunk_size - 1))  # Ensure valid overlap
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Only add non-empty chunks
            chunk_text = ' '.join(chunk_words).strip()
            if chunk_text:  # Only add chunks with actual content
                chunks.append(chunk_text)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def load_pdf(file, filename: str, tags: List[str] = None) -> List[DocumentChunk]:
    """Load and chunk PDF document"""
    chunks = []
    try:
        reader = PdfReader(file)
        full_text = ""
        
        # Extract text from all pages
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""  # Handle None case
            if page_text.strip():  # Only add non-empty pages
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
        
        # Check if we have any text content
        if not full_text.strip():
            st.warning(f"No text content found in PDF: {filename}")
            return chunks
        
        # Chunk the text
        text_chunks = chunk_text(full_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        if not text_chunks:
            st.warning(f"No chunks created from PDF: {filename}")
            return chunks
        
        # Create document chunks
        for i, text_chunk in enumerate(text_chunks):
            page_match = re.search(r'\[Page (\d+)\]', text_chunk)
            page_num = int(page_match.group(1)) if page_match else 1
            
            metadata = {
                "filename": filename,
                "file_type": "pdf",
                "chunk_id": i,
                "page_number": page_num,
                "tags": tags or [],
                "total_chunks": len(text_chunks)
            }
            chunks.append(DocumentChunk(text_chunk, metadata))
    
    except Exception as e:
        st.error(f"Error loading PDF {filename}: {e}")
        # Return empty chunks list on error
    
    return chunks

def load_markdown(file, filename: str, tags: List[str] = None) -> List[DocumentChunk]:
    """Load and chunk Markdown document"""
    chunks = []
    try:
        content = file.read().decode('utf-8')
        
        # Check if we have any content
        if not content.strip():
            st.warning(f"No content found in Markdown file: {filename}")
            return chunks
        
        # Chunk the text
        text_chunks = chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        if not text_chunks:
            st.warning(f"No chunks created from Markdown file: {filename}")
            return chunks
        
        # Create document chunks
        for i, text_chunk in enumerate(text_chunks):
            section = extract_section_header(text_chunk)
            
            metadata = {
                "filename": filename,
                "file_type": "markdown",
                "chunk_id": i,
                "section": section,
                "tags": tags or [],
                "total_chunks": len(text_chunks)
            }
            chunks.append(DocumentChunk(text_chunk, metadata))
    
    except Exception as e:
        st.error(f"Error loading Markdown {filename}: {e}")
        # Return empty chunks list on error
    
    return chunks

def extract_section_header(text: str) -> str:
    """Extract section header from markdown chunk"""
    lines = text.split('\n')
    for line in lines:
        if line.startswith('#'):
            return line.strip()
    return "No section"

# Vector store functions
@st.cache_resource
def init_vector_store():
    """Initialize FAISS vector store"""
    dimension = 768  # Gemini embedding dimension
    index = faiss.IndexFlatIP(dimension)
    return index, []

def get_embeddings(text: str) -> List[float]:
    """Get embeddings for text using Gemini"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return []

def add_documents_to_store(chunks: List[DocumentChunk], index, stored_chunks):
    """Add document chunks to vector store"""
    try:
        embeddings = []
        valid_chunks = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            embedding = get_embeddings(chunk.text)
            if embedding:
                embedding_array = np.array(embedding, dtype=np.float32)
                embedding_array = embedding_array / np.linalg.norm(embedding_array)
                embeddings.append(embedding_array)
                chunk.embedding = embedding_array
                valid_chunks.append(chunk)
        
        progress_bar.empty()
        
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            index.add(embeddings_matrix)
            stored_chunks.extend(valid_chunks)
            return True
        
    except Exception as e:
        st.error(f"Error adding documents: {e}")
    
    return False

def search_documents(query: str, index, stored_chunks, top_k: int = 5):
    """Search for similar chunks"""
    try:
        if index.ntotal == 0:
            return []
        
        query_embedding = get_embeddings(query)
        if not query_embedding:
            return []
        
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_array = query_array / np.linalg.norm(query_array)
        
        scores, indices = index.search(query_array, min(top_k, index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(stored_chunks):
                results.append((stored_chunks[idx], float(score)))
        
        return results
        
    except Exception as e:
        st.error(f"Error searching: {e}")
        return []

def keyword_search(query: str, stored_chunks, top_k: int = 5):
    """Fallback keyword search"""
    query_words = set(query.lower().split())
    results = []
    
    for chunk in stored_chunks:
        chunk_words = set(chunk.text.lower().split())
        common_words = query_words.intersection(chunk_words)
        score = len(common_words) / len(query_words) if query_words else 0
        
        if score > 0:
            results.append((chunk, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def ask_question_with_context(question: str, context: str, conversation_history: str = "") -> str:
    """Ask question with context using Gemini"""
    try:
        model = init_gemini()
        prompt = f"""You are a helpful Q&A assistant. Use the provided context to answer the user's question accurately and concisely.

Context from documents:
{context}

Conversation History:
{conversation_history}

User Question: {question}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide specific details and examples when available
- Keep your response focused and relevant
- If referencing specific information, mention which document it came from

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state
if "vector_index" not in st.session_state:
    st.session_state.vector_index, st.session_state.stored_chunks = init_vector_store()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {}

def display_sources(sources: List[Tuple[DocumentChunk, float]]):
    """Display source information"""
    st.subheader("📚 Sources")
    
    for i, (chunk, score) in enumerate(sources, 1):
        with st.expander(f"Source {i}: {chunk.metadata['filename']} (Score: {score:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                snippet = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                st.text(snippet)
            
            with col2:
                st.write(f"**File:** {chunk.metadata['filename']}")
                st.write(f"**Type:** {chunk.metadata['file_type']}")
                
                if chunk.metadata['file_type'] == 'pdf':
                    st.write(f"**Page:** {chunk.metadata.get('page_number', 'N/A')}")
                elif chunk.metadata['file_type'] == 'markdown':
                    st.write(f"**Section:** {chunk.metadata.get('section', 'N/A')}")
                
                st.write(f"**Relevance Score:** {score:.3f}")

def main():
    st.title("🧠 Q&A Knowledge Base Chatbot")
    st.markdown("Upload documents and ask questions to get AI-powered answers with source references.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📁 Browse Files", "💬 Ask Questions"])
    
    # Upload Tab
    with tab1:
        st.header("Upload Documents")
        st.write("Upload PDF or Markdown files to build your knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'md'],
            help="Upload PDF or Markdown (.md) files"
        )
        
        tags_input = st.text_input(
            "Tags (optional)",
            placeholder="Enter comma-separated tags (e.g., research, documentation, manual)",
            help="Add tags to categorize your documents"
        )
        
        if st.button("Process Files", type="primary"):
            if uploaded_files:
                tag_list = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
                
                with st.spinner("Processing files..."):
                    all_chunks = []
                    upload_results = []
                    
                    for file in uploaded_files:
                        if file.name.endswith('.pdf'):
                            chunks = load_pdf(file, file.name, tag_list)
                        elif file.name.endswith('.md'):
                            chunks = load_markdown(file, file.name, tag_list)
                        else:
                            st.warning(f"Unsupported file type: {file.name}")
                            continue
                        
                        if chunks:
                            all_chunks.extend(chunks)
                            upload_results.append({
                                "filename": file.name,
                                "chunks_created": len(chunks),
                                "tags": tag_list
                            })
                    
                    if all_chunks:
                        success = add_documents_to_store(
                            all_chunks, 
                            st.session_state.vector_index, 
                            st.session_state.stored_chunks
                        )
                        
                        if success:
                            st.success(f"✅ Successfully uploaded {len(uploaded_files)} files")
                            
                            st.subheader("Upload Details")
                            for result in upload_results:
                                st.write(f"- **{result['filename']}**: {result['chunks_created']} chunks created")
                                if result['tags']:
                                    st.write(f"  Tags: {', '.join(result['tags'])}")
                        else:
                            st.error("Failed to process files")
            else:
                st.warning("Please select files to upload.")
    
    # Browse Files Tab
    with tab2:
        st.header("Browse Uploaded Files")
        
        if st.session_state.stored_chunks:
            # Group chunks by filename
            documents = {}
            for chunk in st.session_state.stored_chunks:
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
            
            st.write(f"**Total Documents:** {len(documents)}")
            
            for doc in documents.values():
                with st.expander(f"📄 {doc['filename']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {doc['file_type']}")
                        st.write(f"**Total Chunks:** {doc['total_chunks']}")
                    
                    with col2:
                        if doc['tags']:
                            st.write(f"**Tags:** {', '.join(doc['tags'])}")
                        
                        if doc['file_type'] == 'markdown' and doc['sections']:
                            st.write(f"**Sections:** {', '.join(doc['sections'])}")
        else:
            st.info("No documents uploaded yet. Go to the Upload tab to add documents.")
    
    # Ask Questions Tab
    with tab3:
        st.header("Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            
            for exchange in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(exchange["question"])
                
                with st.chat_message("assistant"):
                    st.write(exchange["answer"])
                    
                    if exchange.get("sources"):
                        with st.expander("View Sources"):
                            display_sources(exchange["sources"])
                
                st.divider()
        
        # Question input
        question = st.text_input(
            "Your Question:",
            placeholder="Ask anything about your uploaded documents...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_button = st.button("Ask Question", type="primary")
        
        with col2:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
        
        if ask_button and question.strip():
            if not st.session_state.stored_chunks:
                st.warning("Please upload some documents first!")
            else:
                with st.spinner("Thinking..."):
                    # Get conversation history
                    session_id = st.session_state.session_id
                    conversation_history = ""
                    if len(st.session_state.chat_history) > 0:
                        recent_exchanges = st.session_state.chat_history[-3:]  # Last 3 exchanges
                        history_parts = []
                        for ex in recent_exchanges:
                            history_parts.append(f"Q: {ex['question']}")
                            history_parts.append(f"A: {ex['answer'][:200]}...")
                        conversation_history = "\n".join(history_parts)
                    
                    # Search for relevant chunks
                    semantic_results = search_documents(
                        question, 
                        st.session_state.vector_index, 
                        st.session_state.stored_chunks, 
                        top_k=5
                    )
                    
                    # Fallback to keyword search if needed
                    if len(semantic_results) < 3:
                        keyword_results = keyword_search(question, st.session_state.stored_chunks, top_k=3)
                        # Combine results
                        all_results = semantic_results + keyword_results
                        seen_chunks = set()
                        unique_results = []
                        for chunk, score in all_results:
                            chunk_id = (chunk.metadata['filename'], chunk.metadata['chunk_id'])
                            if chunk_id not in seen_chunks:
                                seen_chunks.add(chunk_id)
                                unique_results.append((chunk, score))
                        semantic_results = unique_results[:5]
                    
                    if semantic_results:
                        # Prepare context
                        context_parts = []
                        for chunk, score in semantic_results:
                            context_parts.append(f"[Source: {chunk.metadata['filename']}]\n{chunk.text}")
                        
                        context = "\n\n".join(context_parts)
                        
                        # Generate answer
                        answer = ask_question_with_context(question, context, conversation_history)
                        
                        # Add to chat history
                        exchange = {
                            "question": question,
                            "answer": answer,
                            "sources": semantic_results
                        }
                        st.session_state.chat_history.append(exchange)
                        
                        # Display answer
                        st.subheader("🤖 Answer")
                        st.write(answer)
                        
                        # Display sources
                        display_sources(semantic_results)
                        
                        st.rerun()
                    else:
                        st.error("No relevant information found. Try rephrasing your question or upload more documents.")
        
        elif ask_button:
            st.warning("Please enter a question.")
        
        # Show current session info
        with st.sidebar:
            st.subheader("Session Info")
            st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
            st.write(f"**Questions Asked:** {len(st.session_state.chat_history)}")
            st.write(f"**Documents Loaded:** {len(set(chunk.metadata['filename'] for chunk in st.session_state.stored_chunks))}")
            
            if st.button("New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()