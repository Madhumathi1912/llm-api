from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.schemas.models import (
    ChatRequest, ChatResponse,
    ReviewAnalysisRequest, ReviewAnalysisResponse,
    ConversationChatRequest, ConversationChatResponse
)
from app.services.llm_service import LLMService
from app.services.cost_logger import budget_enforcer, BudgetExceededError

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


@router.post("/conversation", response_model=ConversationChatResponse)
async def chat_conversation(request: ConversationChatRequest):
    """
    Multi-turn: omit session_id on your first message to start a new
    conversation. The response includes a session_id — send that same
    value on every following message to continue the SAME conversation
    with full history/context.
    """
    try:
        result = LLMService.ask_with_memory(request.prompt, request.session_id)
    except BudgetExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")
    return ConversationChatResponse(**result)


@router.get("/usage/summary")
async def usage_summary():
    """
    Returns today's total estimated spend and the configured daily limit.
    """
    today_spend = budget_enforcer.get_today_spend()
    return {
        "today_spend_usd": round(today_spend, 6),
        "daily_limit_usd": budget_enforcer.daily_limit_usd,
        "remaining_usd": round(budget_enforcer.daily_limit_usd - today_spend, 6),
    }