from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.schemas.models import (
    ChatRequest, ChatResponse,
    ReviewAnalysisRequest, ReviewAnalysisResponse
)
from app.services.llm_service import LLMService

router = APIRouter()

@router.get("/ping")
async def ping():
    """Sanity check endpoint — confirms router + Swagger wiring works."""
    return {"message": "chat router is alive"}

@router.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest):
    """
    Accepts a single prompt and returns the LLM's response,
    along with token usage for the call.
    """
    try:
        result = LLMService.ask_llm(request.prompt)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")
    return ChatResponse(**result)


@router.post("/reviews/analyze", response_model=ReviewAnalysisResponse)
async def analyze_review(request: ReviewAnalysisRequest):
    """
    Accepts a review text and an optional product name, and returns
    the LLM's analysis of the review, including sentiment, confidence,
    and a suggested reply.
    """
    try:
        result = LLMService.analyze_review(request.review_text, request.product_name, request.temperature)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")
    return ReviewAnalysisResponse(**result)