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

MODERATION_MODEL = "omni-moderation-latest"


class ContentFlaggedError(Exception):
    """
    Raised when input (or output) text is flagged by the moderation
    endpoint. A distinct exception type — same spirit as
    BudgetExceededError — so routers can catch it specifically and
    return a clear, actionable HTTP response rather than a generic 500.
    """
    def __init__(self, flagged_categories: list[str]):
        self.flagged_categories = flagged_categories
        super().__init__(
            f"Content flagged by moderation for: {','.join(flagged_categories)}"
        )


class ModerationClient:
    """
    Thin wrapper around OpenAI's Moderation endpoint. Owns exactly one
    responsibility: classifying text as flagged/not-flagged. Knows
    nothing about chat, RAG, or business logic — callers decide WHAT
    to do when something is flagged (usually: raise ContentFlaggedError
    and stop before the main LLM call).
    """
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key_)


    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def check(self, text: str) -> None:
        """
        Raises ContentFlaggedError if the text is flagged by any
        moderation category. Returns None (no exception) if the text
        is clean — callers just call this and continue if it doesn't raise.
        """
        response = self._client.moderations.create(
            model=MODERATION_MODEL,
            input=text,
        )
        result = response.results[0]
        #Testing Purpose: Log the category scores for debugging and analysis
        category_scores = response.results[0].category_scores
        logger.info(f"Category Scores: {category_scores}")

        if result.flagged:
            flagged_categories = [
                category for category, is_flagged in result.categories.model_dump().items() if is_flagged
            ]
            raise ContentFlaggedError(flagged_categories)


moderation_client = ModerationClient()