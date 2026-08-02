from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAIError

from app.schemas.rag_model import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    RagQueryRequest,
    RagQueryResponse
)
from app.services.rag_service import rag_service
from app.services.text_extractor import UnsupportedFileTypeError
from app.services.moderation_client import ContentFlaggedError
from app.services.tracer import tracer

router = APIRouter()


@router.post('/documents/ingest', response_model=DocumentIngestResponse)
async def ingest(request: DocumentIngestRequest):
    """
    Chunks, embeds, and stores a document's text for later retrieval.
    """
    chunks_created = rag_service.ingest_document(request.text, request.source)
    return DocumentIngestResponse(source=request.source, chunks_created=chunks_created)


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload(
    file: UploadFile = File(...),
    source: str = Form(None),
):
    """
    Accepts an actual file (.txt or .pdf), extracts its text, then
    runs it through the same ingestion pipeline as /documents/ingest.
    'source' is optional — defaults to the uploaded filename if not given.
    """
    filename = file.filename or "uploaded_file"
    file_bytes = await file.read()
 
    try:
        resolved_source, chunks_created = rag_service.ingest_uploaded_file(filename, file_bytes, source)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DocumentIngestResponse(source=resolved_source, chunks_created=chunks_created)


@router.post('/ask', response_model=RagQueryResponse)
async def ask(request: RagQueryRequest):
    """
    Answers a question using retrieval-augmented generation: retrieves
    the most relevant previously-ingested chunks, then asks the LLM to
    answer grounded in that context.
    """
    try:
        result = rag_service.ask(request.question, request.top_k)
    except ContentFlaggedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(exc)}")
    return RagQueryResponse(**result)


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Returns the step-by-step timing breakdown for a given trace_id."""
    steps = tracer.get_trace(trace_id)
    total_ms = sum(s["duration_ms"] for s in steps)
    return {"trace_id": trace_id, "total_duration_ms": round(total_ms, 2), "steps": steps}