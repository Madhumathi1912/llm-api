import hashlib
import json
import logging
import redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)
 
DEFAULT_TTL_SECONDS = 3600  # 1hour

class CacheClient:
    """
    Thin wrapper around Redis. Owns exactly one responsibility: caching
    request/response pairs. Knows nothing about prompts or OpenAI —
    llm_service.py decides WHAT to cache and WHEN; this class just
    handles HOW (key building, serialization, TTL, failure handling).
    """
    def __init__(self):
        self._client = redis.Redis(
            host = settings.redis_host, 
            port = settings.redis_port,
            password = settings.redis_password,
            db = 0,
            decode_responses = True, # returns str instead of bytes
            socket_connect_timeout = 2 # fail fast if Redis is unreachable
        )
    

    @staticmethod
    def build_cache_key(*parts: str) -> str:
        """
        Builds a stable cache key by hashing all given parts together.
        Order matters — callers should always pass parts in the same 
        order (e.g. model, prompt, system_prompt) for consistent hits.
        """
        raw = ":".join(parts)
        return f"llm_cache:{hashlib.sha256(raw.encode()).hexdigest()}"
    
    def get(self, key: str) -> dict | None:
        """
        Returns the cached dict if present, else None.
        Fails OPEN (returns None) if Redis is unreachable — a cache
        outage should degrade to "always call the LLM", not break
        the whole endpoint.
        """
        try:
            raw = self._client.get(key)
        except RedisError as exc:
            logger.warning(f"Cache GET failed, treating as cache miss: {exc}")
            return None
        
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Cache value was not valid JSON, ignoring")
            return None
    
    def set(self, key: str, value: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """
        Stores value as JSON with an expiry. Fails silently (logs only)
        if Redis is unreachable — caching is a nice-to-have, not a
        hard dependency for the request to succeed.
        """
        try:
            self._client.set(key, json.dumps(value), ex=ttl_seconds)
        except RedisError as exc:
            logger.warning(f"Cache SET failed, continuing without caching: {exc}")


# Single shared instance, reused across service modules — same pattern
# as openai_client.py's singleton.
cache_client = CacheClient()    