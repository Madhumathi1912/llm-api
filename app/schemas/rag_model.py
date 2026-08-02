from pydantic import BaseModel, Field

class DocumentIngestRequest(BaseModel):
    text: str = Field(min_length=20, description='The full doc text to ingest')
    source: str = Field(description='A label identifying this doc, example:filename')


class DocumentIngestResponse(BaseModel):
    source: str
    chunks_created: int


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=3, examples=['What is our refund policy'])
    top_k: int = Field(default=3, ge=1, le=10, description='Number of chunks to retrieve as context')


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float


class RagQueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    prompt_tokens: int
    completion_tokens: int
    trace_id: str