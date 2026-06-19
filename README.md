# PalmMind RAG Backend

A production-grade backend for document ingestion and conversational Retrieval-Augmented Generation (RAG). Built with FastAPI, Qdrant, Redis, PostgreSQL, and Groq.

## Features

- **Document Ingestion**: Upload PDF and TXT files, chunk them using Fixed-Size or Semantic strategies.
- **Custom RAG Pipeline**: Fully custom manual RAG pipeline (no RetrievalQAChain) with query rewriting, embedding generation, Qdrant vector search, and context-augmented response generation.
- **Conversational Memory**: Redis-backed chat history for multi-turn conversations.
- **Interview Booking**: LLM-based intent detection and entity extraction to schedule interviews seamlessly during chat.
- **API Key Authentication**: Secure endpoints with API key-based access control.
- **Hexagonal Architecture**: Clean, modular code following industry standards for typing and annotations.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                          │
│  ┌──────────────────┐       ┌──────────────────────────────┐   │
│  │ Document Ingest  │       │    Conversational RAG API    │   │
│  │     API          │       │  ┌────────────────────────┐  │   │
│  │  Upload → Extract│       │  │ Query Rewrite (Groq)   │  │   │
│  │  → Chunk → Embed │       │  │ Embed → Search Qdrant  │  │   │
│  │  → Store         │       │  │ Augment → Generate     │  │   │
│  └──────┬───────────┘       │  │ Booking Detection      │  │   │
│         │                   │  └────────────────────────┘  │   │
│         │                   └──────┬──────────────┬────────┘   │
└─────────┼──────────────────────────┼──────────────┼────────────┘
          │                          │              │
    ┌─────▼─────┐          ┌────────▼──┐    ┌──────▼──────┐
    │  Qdrant   │          │   Redis   │    │ PostgreSQL  │
    │ (Vectors) │          │ (Memory)  │    │ (Metadata)  │
    └───────────┘          └───────────┘    └─────────────┘
```

## Tech Stack

| Component         | Technology                                          |
|-------------------|-----------------------------------------------------|
| **Framework**     | FastAPI                                             |
| **LLM**          | Groq (`llama3-8b-8192`)                              |
| **Embeddings**   | FastEmbed (`BAAI/bge-small-en-v1.5`) — runs locally  |
| **Vector DB**    | Qdrant                                               |
| **Cache/Memory** | Redis                                                |
| **Database**     | PostgreSQL (asyncpg + SQLAlchemy)                    |
| **Migrations**   | Alembic (async)                                      |
| **Doc Parsing**  | PyMuPDF (PDF), chardet (TXT)                         |
| **Auth**         | API Key (`X-API-Key` header)                         |

## Project Structure

```
├── app/
│   ├── api/                    # API route handlers
│   │   ├── ingestion.py        # Document upload/list/delete endpoints
│   │   └── conversation.py     # Chat, history, and booking endpoints
│   ├── auth.py                 # API key authentication dependency
│   ├── config.py               # Settings via pydantic-settings
│   ├── db/
│   │   └── session.py          # Async SQLAlchemy engine & session
│   ├── dependencies.py         # FastAPI dependency injection providers
│   ├── main.py                 # Application factory & lifespan
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── booking.py          # InterviewBooking model
│   │   └── document.py         # Document & DocumentChunk models
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── booking.py
│   │   ├── conversation.py
│   │   └── ingestion.py
│   └── services/               # Business logic layer
│       ├── booking_service.py  # LLM-based booking extraction
│       ├── chat_memory.py      # Redis conversation history
│       ├── chunking.py         # Fixed-size & semantic chunking
│       ├── embedding.py        # FastEmbed wrapper
│       ├── rag_pipeline.py     # Custom RAG orchestration
│       ├── text_extractor.py   # PDF/TXT text extraction
│       └── vector_store.py     # Qdrant operations
├── alembic/                    # Database migrations
├── tests/                      # Unit & integration tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Setup & Installation

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.12+
- [Groq API Key](https://console.groq.com)

### 2. Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `GROQ_API_KEY` — your Groq API key
- `API_KEY` — a secret key for authenticating API requests

### 3. Start Infrastructure

Start Qdrant, Redis, and PostgreSQL using Docker Compose:

```bash
make docker-up
```

### 4. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
make install
```

### 5. Run Database Migrations

```bash
make migrate
```

### 6. Run the Application

```bash
make run
```

The API will be available at `http://localhost:8000`.
Interactive API documentation is available at `http://localhost:8000/docs`.

## Authentication

All API endpoints (except `/health`) require an API key passed via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/documents/
```

## API Usage Examples

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "X-API-Key: your-api-key" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "chunking_strategy=semantic"
```

### 2. Chat (Multi-turn with Booking)

```bash
# First message
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the uploaded document say about PalmMind?"}'

# Follow-up message (use the session_id from the previous response)
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "I want to schedule an interview with Jane Doe (jane@example.com) for 2025-01-15 at 10:00."}'
```

## Testing & Linting

```bash
# Run the test suite
make test

# Run linter
make lint

# Auto-format code
make format
```

## Docker (Full Stack)

Run the entire application stack in Docker:

```bash
make docker-build
docker compose up -d
```

## License

This project is proprietary to Palm Mind Technology.
