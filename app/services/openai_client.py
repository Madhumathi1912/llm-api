import logging
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from app.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError)

class OpenAIClient:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key_)


    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def create_completion(self, **kwargs):
        """
        Calls OpenAI's Chat Completions API with retry logic for transient errors.
        Raises OpenAIError on failure — the router layer decides how to
        translate that into an HTTP response.
        """
        try:
            return self._client.chat.completions.create(**kwargs)
        except OpenAIError as e:
            raise e
        
    
    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def create_structured_completion(self, **kwargs):
        """
        Calls OpenAI's Chat Completions API with structured output and retry logic for transient errors.
        Raises OpenAIError on failure — the router layer decides how to
        translate that into an HTTP response.
        """
        try:
            return self._client.beta.chat.completions.parse(**kwargs)
        except OpenAIError as e:
            raise e
        
        
# Single shared instance — reused across all service modules,
# rather than each one creating its own OpenAI() client.
openai_client = OpenAIClient()