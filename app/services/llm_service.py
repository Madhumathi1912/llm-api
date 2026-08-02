import logging
import uuid

from openai import OpenAI, OpenAIError
from app.config import settings
from app.schemas.models import ReviewAnalysisResponse
from app.services.openai_client import openai_client
from app.services.cache_client import cache_client
from app.services.cost_logger import cost_logger, budget_enforcer, BudgetExceededError
from app.services.conversation_store import conversation_store
from app.services.moderation_client import moderation_client, ContentFlaggedError

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key_)

SYSTEM_PROMPT = "You are a helpful assistant."

class LLMService:

    def ask_llm(prompt: str) -> dict:
        """
        Calls OpenAI's Chat Completions API with a single user prompt.
        Raises OpenAIError on failure — the router layer decides how to
        translate that into an HTTP response.
        """
        try:
            completion = openai_client.create_completion(
                model = settings.openai_model,
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            response = completion.choices[0].message.content
            usage = completion.usage

            return {
                "response": response,
                "model": completion.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        except OpenAIError as e:
            raise e
        

    def analyze_review(review_text: str, product_name: str = None, temperature: float = 0.7) -> dict:
        """
        Calls OpenAI's Chat Completions API to analyze a review and return
        sentiment, confidence, and a suggested reply. Raises OpenAIError on failure.
        """

        SYSTEM_PROMPT = "You are a customer support assistant. Analyze the review and provide sentiment, confidence, and a suggested reply."

        should_cache = temperature == 0 #if temperature is 0 should_cache is True, else False
        cache_key = None
        if should_cache:
            cache_key = cache_client.build_cache_key(settings.openai_model, SYSTEM_PROMPT, review_text, product_name or "")
            cached_result = cache_client.get(cache_key)
            if cached_result is not None:
                cached_result["cached"] = True

                #Log the cost to cost_logger
                print('Adding logs from cache block')
                logger.info('Adding logs from cache block')
                cost_logger.log_usage(
                    endpoint = "/reviews/analyze", 
                    model = cached_result['model'],
                    prompt_tokens = cached_result["prompt_tokens"],
                    completion_tokens = cached_result['completion_tokens'],
                    cached = True
                )
                return cached_result

        try:
            prompt = f"Analyze the following review: '{review_text}'"
            if product_name:
                prompt += f" for the product '{product_name}'."

            logger.info(f"Calling OpenAI structured completion with temperature={temperature}")
            completion = openai_client.create_structured_completion(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=ReviewAnalysisResponse,
            )
        except OpenAIError as e:
            raise e
        
        result = completion.choices[0].message.parsed
        usage = completion.usage
        result = {
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "suggested_reply": result.suggested_reply,
                "model": completion.model, 
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached": False
            }
        
        #Log the cost to cost_logger
        logger.info('Adding logs from non-cache block')
        cost_logger.log_usage(
            endpoint = "/reviews/analyze", 
            model = completion.model,
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            cached = False
        )
        
        #Add the result in cache if should_cache is True
        if should_cache:
            logger.info(f"Caching result for key: {cache_key}")
            cache_client.set(cache_key, result)

        return result


    def ask_with_memory(prompt: str, session_id: str = None) -> dict:
        """
        Multi-turn conversation: builds the message list for this turn
        (system prompt + running summary, if any + recent raw messages),
        sends it to OpenAI, then records this turn — which may trigger
        older messages being folded into the summary if the sliding
        window is exceeded (see ConversationStore).
    
        If session_id is None, a new one is generated — the caller should
        reuse the returned session_id on subsequent messages to continue
        the same conversation.
        """
        moderation_client.check(prompt)
        if not session_id:
            session_id = str(uuid.uuid4())

        messages_for_call = conversation_store.build_messages_for_call(session_id, SYSTEM_PROMPT)
        user_message = {"role": "user", "content": prompt}

        try:
            budget_enforcer.check_budget()
        except BudgetExceededError as e:
            raise e

        try:
            completion = openai_client.create_completion(
                model=settings.openai_model,
                messages=messages_for_call  + [user_message],  # full history + new message
                temperature=0.7,
                max_tokens=300,
            )
        except OpenAIError as e:
            raise e

        assistant_reply = completion.choices[0].message.content
        usage = completion.usage

        assistant_message = {"role": "assistant", "content": assistant_reply}

        updated_data = conversation_store.record_turn(session_id, user_message, assistant_message)

        cost_logger.log_usage(
            endpoint="/chat/conversation",
            model=completion.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached=False,
        )

        return {
            "session_id": session_id,
            "response": assistant_reply,
            "model": completion.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "message_count": len(updated_data['messages']),
        }