# LLM API — FastAPI + OpenAI + RAG

A FastAPI service that wraps the OpenAI API with production-oriented patterns:
retries with backoff, response caching, cost tracking with budget enforcement,
multi-turn conversation memory, function calling, and a full
Retrieval-Augmented Generation (RAG) pipeline backed by a Qdrant vector
database. Fully containerized with Docker Compose.

Built as a hands-on learning project to explore LLM application development
beyond a single API call — reliability patterns, vector search, agentic
tool use, and clean service-layer architecture.

## Features

- **Chat completions** — simple prompt/response endpoint using OpenAI's Chat
  Completions API
- **Structured outputs** — schema-enforced LLM responses (Pydantic models,
  not free-text parsing) for tasks like sentiment analysis
- **Conversation memory** — multi-turn chat with session-based history
  - Sliding window keeps recent messages verbatim
  - Older messages are automatically folded into a running summary
    (via a second LLM call) instead of being dropped outright, once the
    window is exceeded
  - Backed by Redis, with a TTL so abandoned sessions expire
- **Function calling / tool use** — the model can request a real function
  call (e.g. `get_order_status`) with structured arguments; the API executes
  the real Python function and returns the result for a grounded final answer
  - Handles both paths: model answers directly, or requests a tool first
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
- **Testing**
  - Unit tests (`pytest`) covering the deterministic pieces: chunking logic,
    conversation sliding-window/summarization triggering, function-call
    dispatch and two-step orchestration, and retry behavior — all mocked,
    no real API/Redis calls
  - Evals (separate from unit tests) for the non-deterministic LLM-quality
    side: retrieval correctness, answer faithfulness, and LLM-as-judge
    scoring — see `evals/`
- **Clean architecture**
  - Dependency-injected service classes (`OpenAIClient`, `EmbeddingClient`,
    `VectorStore`, `Chunker`, `CostLogger`, `ConversationStore`,
    `ToolCallingService`, `RagService`, etc.)
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
   ┌────────────┬───────────────┼───────────────┬────────────────┐
   │            │               │               │                │
┌──▼──────┐ ┌───▼────────┐ ┌────▼───────┐ ┌─────▼───────┐ ┌──────▼─────────┐
│OpenAI   │ │RagService  │ │Conversation│ │ToolCalling  │ │  CostLogger    │
│Client   │ │(chunk/embed│ │Store       │ │Service      │ │(SQLite usage + │
│(retries)│ │/retrieve/  │ │(sliding    │ │(2-step tool │ │ budgeting)     │
│         │ │ generate)  │ │window +    │ │ call flow)  │ │                │
└──┬──────┘ └─┬────────┬─┘ │summarize)  │ └──────┬──────┘ └────────────────┘
   │          │        │   └─────┬──────┘        │
   │    ┌─────▼──┐ ┌───▼────┐    │           ┌───▼──────────┐
   │    │Chunker+│ │VectorSt│    │           │OrderService  │
   │    │Embed   │ │ore     │    │           │(demo tool    │
   │    │Client  │ │(Qdrant)│    │           │ backend)     │
   │    └────────┘ └───┬────┘    │           └──────────────┘
   │                   │         │
┌──▼──────┐      ┌─────▼───────┐ │
│ OpenAI  │      │   Qdrant    │ └───────────┐
│  API    │      │ (vectors)   │             │
└─────────┘      └─────────────┘       ┌─────▼──────┐
                                       │   Redis    │
                                       │ (caching + │
                                       │  sessions) │
                                       └────────────┘
```

Each service is a single-responsibility class, constructed with its
dependencies injected rather than importing globals directly — this keeps
business logic decoupled from the infrastructure it depends on, and makes
mocking straightforward for unit tests.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI, Uvicorn |
| LLM | OpenAI API (Chat Completions, Embeddings, Function Calling) |
| Vector database | Qdrant |
| Caching & session store | Redis |
| Cost tracking | SQLite |
| Retry logic | Tenacity |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| Validation | Pydantic |

## Project Structure

```
app/
├── main.py                        # FastAPI app instance, router registration
├── config.py                      # Environment-based settings (Pydantic Settings)
├── routers/
│   ├── chat.py                     # /chat/* endpoints (single-turn + conversation)
│   ├── rag.py                       # /rag/* endpoints (ingest, upload, ask)
│   └── tools.py                      # /chat/tools/* endpoint (function calling)
├── schemas/
│   ├── models.py                     # Request/response models for chat + conversation
│   ├── rag_model.py                       # Request/response models for RAG endpoints
│   └── tool_model.py                      # Request/response models for tool-calling endpoint
└── services/
    ├── openai_client.py             # Retry-wrapped OpenAI chat completions client
    ├── embedding_client.py           # Retry-wrapped OpenAI embeddings client
    ├── vector_store.py                # Qdrant wrapper (storage + similarity search)
    ├── chunking.py                     # Text chunking (Chunker class)
    ├── text_extractor.py                # File text extraction (.txt/.pdf/.docx)
    ├── rag_service.py                    # Orchestrates the full RAG pipeline
    ├── cache_client.py                    # Redis wrapper for response caching
    ├── cost_logger.py                      # SQLite usage logging + budget enforcement
    ├── conversation_store.py                # Redis-backed session history, sliding window
    ├── conversation_summarizer.py             # Condenses overflow messages into a summary
    ├── order_service.py                        # Demo backend for the function-calling tool
    └── tool_calling_service.py                  # Two-step function-calling orchestration

tests/                              # Unit tests (mocked, no real API/Redis calls)
├── test_chunking.py
├── test_conversation_store.py
├── test_openai_client.py
└── test_tool_calling_service.py

evals/                              # LLM-quality evals (separate from unit tests)
└── ...
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
- `redis` — response cache + conversation session store (port `6379`)
- `qdrant` — vector database (ports `6333`/`6334`, dashboard at `6333/dashboard`)

### 3. Open the interactive API docs

```
http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat/` | Single-turn prompt → LLM response (no memory) |
| POST | `/chat/conversation` | Multi-turn chat with session-based memory |
| POST | `/chat/tools/` | Chat with function-calling — model can call `get_order_status` |
| GET | `/chat/usage/summary` | Today's estimated spend vs. daily budget |
| POST | `/rag/documents/ingest` | Ingest raw text into the vector store |
| POST | `/rag/documents/upload` | Upload a `.txt`/`.pdf`/`.docx` file to ingest |
| POST | `/rag/ask` | Ask a question, answered using retrieved context |

### Using conversation memory

Omit `session_id` on your first message to start a new conversation — the
response includes a `session_id` to reuse on every following message to
continue the same conversation with full context.

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Unit tests cover deterministic logic only (chunking, sliding-window/
summarization triggering, tool-call dispatch, retry behavior) using mocks —
no real OpenAI, Redis, or Qdrant calls are made during test runs.

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
- Conversation memory uses a message-count sliding window, not a
  token-count based one — a natural refinement for more precise context
  budgeting
- No streaming responses yet — replies are returned in full, not
  token-by-token
- No authentication on the API itself
- `.doc` (legacy binary Word format) is not supported — only `.docx`
- Function calling currently supports a single demo tool
  (`get_order_status`) backed by in-memory fake data

## License

Personal learning project — no license restrictions.