from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, description="The prompt to send to the LLM.", examples=["What is the capital of France?"])

class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int

class ReviewAnalysisRequest(BaseModel):
    review_text: str = Field(min_length=5, max_length=50, description="The text of the review to analyze.", examples=["This product is amazing! I loved it."])
    product_name: Optional[str] = Field(None, description="The name of the product being reviewed.", examples=["Apple iPhone 13"])
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature for the LLM response. Lower values make the output more deterministic.", examples=[0.0, 0.5, 1.0])

class ReviewAnalysisResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    suggested_reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool