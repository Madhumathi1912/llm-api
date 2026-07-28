import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import FastAPI
from app.routers import chat, rag

app = FastAPI(
    title="LLM API",
    description="A simple FastAPI wrapper around an LLM provider.",
    version="0.1.0"
)

app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(rag.router, prefix="/rag", tags=['RAG'])

@app.get('/health', tags=["Health"])
async def health_check():
    return {"status": "ok"}