from pydantic import BaseModel, Field


class ToolChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["What's the status of order ORD1001?"])


class ToolChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    tool_calls_made: list[str] # names of functions actually called — empty if none were needed