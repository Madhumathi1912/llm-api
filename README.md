# LLM API — FastAPI + OpenAI + RAG

A FastAPI service that wraps the OpenAI API with production-oriented patterns:
retries with backoff, response caching, cost tracking with budget enforcement,
and a full Retrieval-Augmented Generation (RAG) pipeline backed by a Qdrant
vector database. Fully containerized with Docker Compose.

Built as a hands-on learning project to explore LLM application development
beyond a single API call — reliability patterns, vector search, and clean
service-layer architecture.

## Features

- **Chat completions** — simple prompt/response endpoint using OpenAI's Chat
  Completions API
- **Structured outputs** — schema-enforced LLM responses (Pydantic models,
  not free-text parsing) for tasks like sentiment analysis
- **Retrieval-Augmented Generation (RAG)**
  - Document ingestion from raw text or uploaded files (`.txt`, `.pdf`, `.docx`)
  - Automatic chunking, embedding (OpenAI `text-embedding-3-small`), and
    storage in Qdrant
  - Similarity search + context-grounded question answering, with source
    chunks returned alongside every answer
- **Reliability**
  - Automatic retries with exponential backoff (`tenacity`) on transient
    OpenAI failures (rate limits, timeouts, connection errors)
  - Fail-open Redis caching for deterministic (`temperature=0`) requests
- **Cost tracking & budgeting**
  - Every call logged to SQLite with an estimated cost, based on OpenAI's
    per-model pricing
  - Configurable daily spending limit — requests are rejected (`429`) once
    the limit is reached
- **Clean architecture**
  - Dependency-injected service classes (`OpenAIClient`, `EmbeddingClient`,
    `VectorStore`, `Chunker`, `CostLogger`, `RagService`, etc.)
  - Each service owns one responsibility and depends only on the public
    interface of its collaborators — easy to unit test with mocks, easy to
    swap implementations (e.g. a different vector DB or LLM provider)

## Architecture

```
                         ┌─────────────┐
                         │   FastAPI   │
                         │   (app)     │
                         └──────┬──────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼─────────┐    ┌─────────▼──────────┐    ┌─────────▼────────┐
│  OpenAIClient   │    │    RagService      │    │   CostLogger     │
│  (retries via   │    │  (orchestrates     │    │  (SQLite usage   │
│   tenacity)     │    │   ingestion & Q&A) │    │   + budgeting)   │
└───────┬─────────┘    └─────┬───────┬──────┘    └──────────────────┘
        │                    │       │
        │           ┌────────▼──┐ ┌──▼───────────────┐
        │           │ Chunker + │ │  VectorStore     │
        │           │ Embedding │ │  (Qdrant client) │
        │           │  Client   │ └────────┬─────────┘
        │           └───────────┘          │
        │                                  │
┌───────▼─────────┐              ┌─────────▼───────────┐    ┌─────────────┐
│   OpenAI API    │              │      Qdrant         │    │    Redis    │
│ (chat + embed)  │              │  (vector database)  │    │  (caching)  │
└─────────────────┘              └─────────────────────┘    └─────────────┘
```

Each box under `app` is a single-responsibility service class, constructed
with its dependencies injected rather than importing globals directly —
this keeps the business logic (`RagService`) decoupled from the
infrastructure it depends on.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI, Uvicorn |
| LLM | OpenAI API (Chat Completions, Embeddings) |
| Vector database | Qdrant |
| Caching | Redis |
| Cost tracking | SQLite |
| Retry logic | Tenacity |
| Containerization | Docker, Docker Compose |
| Validation | Pydantic |

## Project Structure

```
app/
├── main.py                    # FastAPI app instance, router registration
├── config.py                  # Environment-based settings (Pydantic Settings)
├── routers/
│   ├── chat.py                 # /chat/* endpoints
│   └── rag.py                  # /rag/* endpoints (ingest, upload, ask)
├── schemas/
│   ├── chat.py                 # Request/response models for chat endpoints
│   └── rag.py                  # Request/response models for RAG endpoints
└── services/
    ├── openai_client.py         # Retry-wrapped OpenAI chat completions client
    ├── embedding_client.py       # Retry-wrapped OpenAI embeddings client
    ├── vector_store.py           # Qdrant wrapper (storage + similarity search)
    ├── chunking.py                # Text chunking (Chunker class)
    ├── text_extractor.py          # File text extraction (.txt/.pdf/.docx)
    ├── rag_service.py              # Orchestrates the full RAG pipeline
    ├── cache_client.py              # Redis wrapper for response caching
    └── cost_logger.py                # SQLite usage logging + budget enforcement
```

## Setup

### Prerequisites

- Docker Desktop
- An OpenAI API key

### 1. Clone and configure environment variables

```bash
git clone https://github.com/Madhumathi1912/llm-api.git
cd llm-api
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```dotenv
OPENAI_API_KEY_=sk-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
```

> Note: the trailing underscore in `OPENAI_API_KEY_` is intentional — it
> avoids collisions with a machine-wide `OPENAI_API_KEY` environment
> variable some systems may already have set.

### 2. Start everything with Docker Compose

```bash
docker compose up --build
```

This starts three services:
- `app` — the FastAPI application (port `8000`)
- `redis` — response cache (port `6379`)
- `qdrant` — vector database (ports `6333`/`6334`, dashboard at `6333/dashboard`)

### 3. Open the interactive API docs

```
http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat/` | Simple prompt → LLM response |
| GET | `/chat/usage/summary` | Today's estimated spend vs. daily budget |
| POST | `/rag/documents/ingest` | Ingest raw text into the vector store |
| POST | `/rag/documents/upload` | Upload a `.txt`/`.pdf`/`.docx` file to ingest |
| POST | `/rag/ask` | Ask a question, answered using retrieved context |

## Running Locally Without Docker

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

You'll need Redis and Qdrant running separately (e.g. via Docker) and
`REDIS_HOST`/`QDRANT_HOST` in `.env` set to `localhost`.

## Known Limitations / Next Steps

- Chunking is fixed-size (character-based) rather than semantic — long
  bulleted/list-style content can occasionally be split across chunk
  boundaries, affecting retrieval for "list all X" style questions unless
  `top_k` is increased
- No conversation memory yet — every request is single-turn/stateless
- No streaming responses — replies are returned in full, not token-by-token
- No authentication on the API itself
- `.doc` (legacy binary Word format) is not supported — only `.docx`

## License

Personal learning project — no license restrictions.