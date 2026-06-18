# PalmMind AI Backend

A production-grade backend for document ingestion and conversational Retrieval-Augmented Generation (RAG). Built using FastAPI, Qdrant, Redis, PostgreSQL, and Google Gemini.

## Features
- **Document Ingestion**: Upload PDF and TXT files, chunk them using Fixed-Size or Semantic strategies.
- **Custom RAG Pipeline**: Fully custom manual RAG pipeline (no RetrievalQAChain) with query rewriting, embedding generation, Qdrant vector search, and context-augmented response generation.
- **Conversational Memory**: Redis-backed chat history for multi-turn conversations.
- **Interview Booking**: LLM-based intent detection and entity extraction to schedule interviews seamlessly during chat.
- **Hexagonal Architecture**: Clean, modular code following industry standards for typing and annotations.

## Tech Stack
- **Framework**: FastAPI
- **LLM & Embeddings**: Google Gemini (`gemini-2.0-flash` and `text-embedding-004`)
- **Vector Database**: Qdrant
- **Cache/Memory**: Redis
- **Database**: PostgreSQL (with asyncpg & SQLAlchemy)
- **Document Parsing**: PyMuPDF (PDF), chardet (TXT)

## Setup & Installation

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Google Gemini API Key

### 2. Environment Variables
Copy the example environment file and add your Gemini API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 3. Start Infrastructure
Start Qdrant, Redis, and PostgreSQL using Docker Compose:
```bash
make docker-up
```

### 4. Install Dependencies
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate
make install
```

### 5. Run the Application
```bash
make run
```
The API will be available at `http://localhost:8000`.
Interactive API documentation is available at `http://localhost:8000/docs`.

## Testing & Linting
Run the test suite:
```bash
make test
```

Run linting and formatting:
```bash
make lint
make format
```

## API Usage Examples

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "strategy=semantic"
```

### 2. Chat (Multi-turn with Booking)
```bash
# First message
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, what does the uploaded document say about PalmMind?"}'

# Second message (booking intent)
# Note: Use the session_id returned from the previous response
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "message": "I want to schedule an interview with Jane Doe (jane@example.com) for tomorrow at 10:00."}'
```
