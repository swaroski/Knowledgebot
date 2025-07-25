from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import uuid

from fastapi_app.document_loader import document_loader
from fastapi_app.vector_store import vector_store
from fastapi_app.gemini_client import gemini_client
from fastapi_app.conversation import conversation_memory

app = FastAPI(title="Q&A Knowledge Base API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    tags: Optional[str] = Form(None)
):
    """Upload multiple PDF or Markdown documents"""
    try:
        uploaded_files = []
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        for file in files:
            if not file.filename:
                continue
                
            # Validate file type
            if not (file.filename.endswith('.pdf') or file.filename.endswith('.md')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only PDF and Markdown files are supported."
                )
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process file based on type
                if file.filename.endswith('.pdf'):
                    chunks = document_loader.load_pdf(tmp_file_path, file.filename, tag_list)
                else:  # .md file
                    chunks = document_loader.load_markdown(tmp_file_path, file.filename, tag_list)
                
                if chunks:
                    success = vector_store.add_documents(chunks)
                    if success:
                        uploaded_files.append({
                            "filename": file.filename,
                            "chunks_created": len(chunks),
                            "tags": tag_list
                        })
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}")
                else:
                    raise HTTPException(status_code=500, detail=f"No content extracted from {file.filename}")
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer with sources"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = conversation_memory.get_conversation_history(session_id)
        
        # Search for relevant chunks (semantic search)
        semantic_results = vector_store.search(request.question, top_k=5)
        
        # Fallback to keyword search if semantic search returns few results
        if len(semantic_results) < 3:
            keyword_results = vector_store.keyword_search(request.question, top_k=3)
            # Combine and deduplicate results
            all_results = semantic_results + keyword_results
            seen_chunks = set()
            unique_results = []
            for chunk, score in all_results:
                chunk_id = (chunk.metadata['filename'], chunk.metadata['chunk_id'])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append((chunk, score))
            semantic_results = unique_results[:5]
        
        if not semantic_results:
            raise HTTPException(status_code=404, detail="No relevant information found in the knowledge base")
        
        # Prepare context and sources
        context_parts = []
        sources = []
        
        for chunk, score in semantic_results:
            context_parts.append(f"[Source: {chunk.metadata['filename']}]\n{chunk.text}")
            
            source_info = {
                "filename": chunk.metadata['filename'],
                "file_type": chunk.metadata['file_type'],
                "chunk_id": chunk.metadata['chunk_id'],
                "score": score,
                "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            
            if chunk.metadata['file_type'] == 'pdf':
                source_info["page_number"] = chunk.metadata.get('page_number', 1)
            elif chunk.metadata['file_type'] == 'markdown':
                source_info["section"] = chunk.metadata.get('section', 'No section')
            
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using Gemini
        answer = gemini_client.ask_question(request.question, context, conversation_history)
        
        # Store in conversation memory
        conversation_memory.add_exchange(session_id, request.question, answer, sources)
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs")
async def get_documents():
    """Get list of uploaded documents with metadata"""
    try:
        documents = vector_store.get_all_documents()
        return {
            "total_documents": len(documents),
            "documents": list(documents.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Q&A Knowledge Base API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Q&A Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDF or Markdown documents",
            "ask": "POST /ask - Ask questions about uploaded documents",
            "docs": "GET /docs - List uploaded documents",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)