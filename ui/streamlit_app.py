import streamlit as st
import requests
import json
from typing import List, Dict, Any
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="Q&A Knowledge Base",
    page_icon="🧠",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def upload_files(files, tags):
    """Upload files to the API"""
    try:
        file_data = []
        for file in files:
            file_data.append(("files", (file.name, file.read(), file.type)))
        
        # Reset file pointers
        for file in files:
            file.seek(0)
        
        data = {"tags": tags} if tags else {}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=file_data,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def ask_question(question: str, session_id: str):
    """Ask a question to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "session_id": session_id}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get documents: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return None

def display_sources(sources: List[Dict[str, Any]]):
    """Display source information"""
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source['filename']} (Score: {source['score']:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text(source['snippet'])
            
            with col2:
                st.write(f"**File:** {source['filename']}")
                st.write(f"**Type:** {source['file_type']}")
                
                if source['file_type'] == 'pdf':
                    st.write(f"**Page:** {source.get('page_number', 'N/A')}")
                elif source['file_type'] == 'markdown':
                    st.write(f"**Section:** {source.get('section', 'N/A')}")
                
                st.write(f"**Relevance Score:** {source['score']:.3f}")

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
        
        if st.button("Upload Files", type="primary"):
            if uploaded_files:
                with st.spinner("Uploading and processing files..."):
                    result = upload_files(uploaded_files, tags_input)
                    
                if result:
                    st.success(f"✅ {result['message']}")
                    
                    # Display upload details
                    st.subheader("Upload Details")
                    for file_info in result['files']:
                        st.write(f"- **{file_info['filename']}**: {file_info['chunks_created']} chunks created")
                        if file_info['tags']:
                            st.write(f"  Tags: {', '.join(file_info['tags'])}")
            else:
                st.warning("Please select files to upload.")
    
    # Browse Files Tab
    with tab2:
        st.header("Browse Uploaded Files")
        
        if st.button("Refresh File List"):
            st.rerun()
        
        with st.spinner("Loading documents..."):
            docs_data = get_documents()
        
        if docs_data:
            st.write(f"**Total Documents:** {docs_data['total_documents']}")
            
            if docs_data['documents']:
                for doc in docs_data['documents']:
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
        else:
            st.error("Failed to load documents.")
    
    # Ask Questions Tab
    with tab3:
        st.header("Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            
            for i, exchange in enumerate(st.session_state.chat_history):
                # User question
                with st.chat_message("user"):
                    st.write(exchange["question"])
                
                # AI answer
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
            with st.spinner("Thinking..."):
                result = ask_question(question, st.session_state.session_id)
            
            if result:
                # Add to chat history
                exchange = {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
                st.session_state.chat_history.append(exchange)
                
                # Update session ID
                st.session_state.session_id = result["session_id"]
                
                # Display the new answer
                st.subheader("🤖 Answer")
                st.write(result["answer"])
                
                # Display sources
                if result["sources"]:
                    display_sources(result["sources"])
                
                # Clear input
                st.rerun()
        
        elif ask_button:
            st.warning("Please enter a question.")
        
        # Show current session info
        with st.sidebar:
            st.subheader("Session Info")
            st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
            st.write(f"**Questions Asked:** {len(st.session_state.chat_history)}")
            
            if st.button("New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()