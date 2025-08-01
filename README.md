# Q&A Knowledge Base Chatbot

A powerful Q&A chatbot that allows you to upload PDF and Markdown documents, then ask questions and get AI-powered answers with source references. Built with Streamlit and Google's Gemini API.

> **🚀 Quick Start**: Use the standalone `streamlit_app.py` in the root directory for the easiest deployment experience!

## Features

- **Multi-format Document Support**: Upload PDF and Markdown files
- **Intelligent Chunking**: Automatically splits documents into semantic chunks with overlap
- **Semantic Search**: Uses Gemini embeddings with FAISS for fast similarity search
- **Keyword Fallback**: Automatic fallback to keyword search when semantic search yields few results
- **Conversational Memory**: Maintains context across follow-up questions per session
- **Source Transparency**: Shows relevant document snippets with filenames and page/section references
- **Document Management**: Tag and browse uploaded documents
- **Chat Interface**: User-friendly Streamlit UI with conversation history

## Tech Stack

- **Frontend**: Streamlit (standalone application)
- **AI**: Google Gemini API (embedding-001 and gemini-2.5-pro)
- **Vector DB**: FAISS (in-memory)
- **Document Processing**: pypdf for PDFs, built-in markdown parsing
- **Python**: 3.10+

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd knowledgebot
pip install -r requirements.txt
```

### 2. Configure Environment

**For Local Development:**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**For Streamlit Cloud Deployment:**
Configure your API key in `.streamlit/secrets.toml`

Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Run the Application

**Standalone Streamlit App (Recommended):**
```bash
streamlit run streamlit_app.py
```

**Alternative - FastAPI Backend + Separate Frontend:**
```bash
# Terminal 1 - Start FastAPI Backend:
uvicorn fastapi_app.main:app --reload

# Terminal 2 - Start original Streamlit Frontend:
streamlit run ui/streamlit_app.py  # (if ui folder exists)
```

### 4. Access the Application

- **Standalone Streamlit**: http://localhost:8501
- **FastAPI Docs** (if using alternative setup): http://localhost:8000/docs

## Usage

### Upload Documents
1. Go to the "Upload" tab
2. Select PDF or Markdown files
3. Add optional tags for categorization
4. Click "Process Files"

### Browse Documents
1. Go to "Browse Files" tab to see all uploaded documents
2. View file metadata, tags, and sections

### Ask Questions
1. Go to "Ask Questions" tab
2. Type your question in the input field
3. View AI-generated answers with source references
4. Ask follow-up questions for contextual conversations

## Configuration

### Environment Variables (.env file):

```bash
GEMINI_API_KEY=your_api_key_here
FAISS_INDEX_PATH=faiss_index.bin  # Path to FAISS index file (optional)
CHUNK_SIZE=500                    # Document chunk size in tokens
CHUNK_OVERLAP=50                  # Overlap between chunks
MAX_CONTEXTS=5                    # Max contexts sent to AI
```

### Streamlit Secrets (.streamlit/secrets.toml):

```toml
[general]
GEMINI_API_KEY = "your_gemini_api_key_here"
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CONTEXTS = 5
```

## Project Structure

```
knowledgebot/
├── streamlit_app.py       # 🎯 Main standalone Streamlit application
├── fastapi_app/           # Optional FastAPI backend (alternative)
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── document_loader.py  # PDF/Markdown parsing & chunking
│   ├── vector_store.py     # FAISS vector operations
│   ├── gemini_client.py    # Gemini API integration
│   ├── conversation.py     # Chat memory management
│   └── config.py          # Configuration management
├── .streamlit/
│   └── secrets.toml       # Streamlit secrets for cloud deployment
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template for local development
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

## Key Features Detail

### Semantic + Keyword Search
- **Primary**: Uses Gemini embeddings with FAISS for semantic similarity
- **Fallback**: Keyword-based search when semantic results are insufficient  
- **Combined**: Merges both approaches for comprehensive retrieval

### Follow-up Questions
- Maintains conversation context per session
- Includes previous Q&A in context for better follow-up understanding
- Session-based memory management

### Source Attribution
- Shows document snippets with relevance scores
- Includes filename, page numbers (PDF), or sections (Markdown)
- Transparent source referencing for all answers

### Document Processing
- **PDF**: Extracts text with page number tracking
- **Markdown**: Preserves section headers and structure
- **Intelligent chunking**: Configurable size and overlap with validation

## Deployment Options

### Streamlit Cloud (Recommended)
1. Fork/clone this repository
2. Deploy `streamlit_app.py` to Streamlit Cloud
3. Configure secrets in your Streamlit Cloud dashboard
4. Add your `GEMINI_API_KEY` to secrets

### Local Development
```bash
streamlit run streamlit_app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**
   - Ensure your `.env` file contains a valid Gemini API key (local)
   - Or configure secrets in `.streamlit/secrets.toml` (cloud)

2. **"No relevant information found"**
   - Upload documents first
   - Try rephrasing your question
   - Check if documents were processed successfully

3. **PDF processing errors**
   - Some PDFs may have no extractable text (scanned images)
   - Try using OCR tools first, or use different PDF files

4. **Connection/import errors**
   - Use the standalone `streamlit_app.py` (recommended)
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

### API Key Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your configuration:
   - **Local**: `.env` file as `GEMINI_API_KEY=your_key_here`
   - **Cloud**: Streamlit secrets as shown above

## Development

### Running Tests
```bash
# No tests included in this minimal version
# Add your own tests as needed
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use this project for your own applications.

## Changelog

- **v1.0**: Initial release with standalone Streamlit app
- **v1.1**: Added FastAPI backend option
- **v1.2**: Improved error handling and PDF processing
- **v1.3**: Enhanced UI and added better documentation