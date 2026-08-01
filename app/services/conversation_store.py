import json
import logging
import redis
from redis.exceptions import RedisError

from app.config import settings
from app.services.conversation_summarizer import ConversationSummarizer

from app.services.openai_client import openai_client

logger = logging.getLogger(__name__)

CONVERSATION_TTL_SECONDS = 3600  # abandoned conversations expire after 1 hour

# How many of the MOST RECENT raw messages to keep verbatim(exactly as it is typed). Once a
# conversation exceeds this, the OLDEST excess messages get folded into
# the running summary instead of being kept raw or dropped entirely.
SLIDING_WINDOW_SIZE = 4

class ConversationStore:
    """storing and retrieving conversation message history, per session, in Redis"""
    def __init__(self, summarizer_: ConversationSummarizer):
        self._summarizer = summarizer_
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=0,
            decode_responses=True,
        )


    def _key(self, session_id: str) -> str:
        return f"conversation:{session_id}"


    def _load(self, session_id: str) -> dict:
        """
        Returns {"summary": str, "messages": list[dict]} for this session,
        or a fresh empty structure if it doesn't exist yet. Fails open on
        any Redis/JSON error — a lost history degrades gracefully rather
        than breaking the request.
        """
        try:
            raw = self._client.get(self._key(session_id))
        except RedisError as exc:
            logger.warning(f"ConversationStore GET failed, starting fresh: {exc}")
            return {"summary": "", "messages": []}
        if raw is None:
            return {'summary': "", 'messages': []}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Stored conversation was not valid JSON, starting fresh")
            return {"summary": "", "messages": []}


    def _save(self, session_id: str, data: dict) -> None:
        try:
            self._client.set(
                self._key(session_id),
                json.dumps(data),
                ex=CONVERSATION_TTL_SECONDS #Expiration
            )
        except RedisError as exc:
            logger.warning(f"ConversationStore SET failed, history won't persist: {exc}")


    def build_messages_for_call(self, session_id: str, system_prompt: str) -> list[dict]:
        """
        Builds the full message list to send to OpenAI for this turn:
        [system prompt] + [summary, if one exists] + [recent raw messages].
        The system prompt is passed in fresh each call rather than stored
        — it's static config, not conversation state, so there's no risk
        of it being dropped by any truncation logic.
        """
        data = self._load(session_id)
        messages = [{'role': 'system', 'content': system_prompt}]
        if data['summary']:
            messages.append({
                'role': 'system',
                'content': f'Summary of earlier conversation: {data['summary']}',
            })
            messages.extend(data['messages'])
        return messages


    def record_turn(self, session_id: str, user_message: dict, assistant_message: dict) -> dict:
        """
        Adds this turn's user+assistant messages to the sliding window.
        If that pushes the window over SLIDING_WINDOW_SIZE, the OLDEST
        excess messages are folded into the running summary via
        ConversationSummarizer, then dropped from the raw list — their
        content survives in condensed form, not lost entirely.

        Returns the saved data ({"summary", "messages"}) so the caller
        can report things like message_count.
        """
        data = self._load(session_id)
        data['messages'].extend([user_message, assistant_message])

        if len(data['messages']) > SLIDING_WINDOW_SIZE:
            overflow_count = len(data['messages']) - SLIDING_WINDOW_SIZE
            overflow_messages = data['messages'][:overflow_count]
            data['messages'] = data['messages'][overflow_count:]

            data['summary'] = self._summarizer.summarize(
                data['summary'], overflow_messages
            )
            logger.info(
                f"Session {session_id}: folded {len(overflow_messages)} "
                f"messages into summary, kept last {SLIDING_WINDOW_SIZE} raw"
            )
        self._save(session_id, data)
        return data


    def clear(self, session_id: str) -> None:
        """Deletes a session's history entirely — useful for a 'reset chat' action."""
        try:
            self._client.delete(self._key(session_id))
        except RedisError as exc:
            logger.warning(f"ConversationStore DELETE failed: {e}")


# Single shared instance, built with its dependency injected — same
# pattern as rag_service.py's construction at the bottom of that file.
conversation_store = ConversationStore(
    summarizer_=ConversationSummarizer(openai_client_=openai_client)
)