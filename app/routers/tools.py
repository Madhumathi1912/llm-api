from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.schemas.tool_model import ToolChatRequest, ToolChatResponse
from app.services.tool_calling_service import tool_calling_service
from app.services.moderation_client import ContentFlaggedError

router = APIRouter()


@router.post("/", response_model=ToolChatResponse)
async def chat_with_tools(request: ToolChatRequest):
    """
    Accepts a prompt. The model decides whether it needs to call
    get_order_status to answer — if so, it's actually executed against
    OrderService's real (fake, for this demo) data, and the final
    answer is grounded in that real result. tool_calls_made will be
    empty if the model answered without needing any tool.
    """
    try:
        result = tool_calling_service.ask(request.prompt)
    except ContentFlaggedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")

    return ToolChatResponse(**result)