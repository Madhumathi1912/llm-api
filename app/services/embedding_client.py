import logging
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from app.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError)
# text-embedding-3-small produces 1536-dimensional vectors. Needed when
# creating the Qdrant collection, since Qdrant requires the vector size
# to be fixed upfront per collection.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

class EmbeddingClient:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key_)


    @retry(
            retry = retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            wait = wait_exponential(multiplier=1, min=1, max=10),
            stop = stop_after_attempt(3),
            before_sleep = before_sleep_log(logger, logging.WARNING),
            reraise = True,
    )
    def embed(self, text: str) -> list[float]:
        """
        Returns a single embedding vector for the given text.
        Raises OpenAIError on failure (after retries exhausted).
        """
        response = self._client.embeddings.create(
            model = EMBEDDING_MODEL,
            input = text
        )
        return response.data[0].embedding
    

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True

    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Turning each chunk into numbers
        Embeds multiple texts in a SINGLE API call — much more efficient
        than calling embed() in a loop when indexing many chunks at once.
        Returns vectors in the same order as the input texts.
        """
        response = self._client.embeddings.create(
            model = EMBEDDING_MODEL,
            input = texts,
        )
        return [item.embedding for item in response.data]
         

# Single shared instance
embedding_client = EmbeddingClient()