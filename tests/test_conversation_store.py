from unittest.mock import MagicMock, patch
import json

from app.services.conversation_store import ConversationStore


def _make_store_with_fake_redis():
    """
    Builds a ConversationStore backed by an in-memory fake instead of a
    real Redis connection. We patch redis.Redis itself so __init__
    never actually tries to connect anywhere.
    """
    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "Fake summary of earlier turns"

    with patch("app.services.conversation_store.redis.Redis") as MockRedis:
        # Simulate Redis with a simple in-memory dict
        fake_storage = {}

        def fake_get(key):
            return fake_storage.get(key)

        def fake_set(key, value, ex=None):
            fake_storage[key] = value

        mock_redis_instance = MagicMock()
        mock_redis_instance.get.side_effect = fake_get
        mock_redis_instance.set.side_effect = fake_set
        MockRedis.return_value = mock_redis_instance

        store = ConversationStore(summarizer_=fake_summarizer)
        return store, fake_storage, fake_summarizer


def test_new_session_has_no_history():
    store, _, _ = _make_store_with_fake_redis()
    messages = store.build_messages_for_call("brand-new-session", system_prompt="You're helpful")
    # Only the system prompt should be present — no summary, no prior turns
    assert messages == [{'role': 'system', 'content': "You're helpful"}]


def test_record_turn_stores_messages():
    store, fake_storage, _ = _make_store_with_fake_redis()
    session_id = "session-1"

    store.record_turn(
        session_id,
        user_message={"role": "user", "content": "Hello"},
        assistant_message={"role": "assistant", "content": "Hi there!"},
    )
    stored = json.loads(fake_storage[f"conversation:{session_id}"])
    assert len(stored['messages']) == 2
    assert stored['messages'][0]['content'] == "Hello"


def test_sliding_window_triggers_summarization_when_exceeded():
    """
    SLIDING_WINDOW_SIZE is 4. Adding 3 turns (6 messages total) should
    exceed it, causing the oldest messages to be folded into a summary
    via the (mocked) summarizer, rather than growing unbounded.
    """
    store, fake_storage, fake_summarizer = _make_store_with_fake_redis()
    session_id = "session-2"

    for i in range(3): # 3 turns = 6 messages, window size is 4
        store.record_turn(
            session_id,
            user_message={"role": "user", "content": f"Message {i}"},
            assistant_message={"role": "assistant", "content": f"Reply {i}"},
        )
    stored = json.loads(fake_storage[f"conversation:{session_id}"])

    # Raw messages should be capped at the window size
    assert len(stored["messages"]) <= 4
    # Summarizer should have been called at least once, since we exceeded the window
    assert fake_summarizer.summarize.called
    # The summary text should now be present in stored data
    assert stored["summary"] == "Fake summary of earlier turns"


def test_summary_included_in_messages_for_call_once_it_exists():
    store, fake_storage, fake_summarizer = _make_store_with_fake_redis()
    session_id = "session-3"

    for i in range(3):
        store.record_turn(
            session_id,
            {"role": "user", "content": f"Message {i}"},
            {"role": "assistant", "content": f"Reply {i}"},
        )

    messages = store.build_messages_for_call(session_id, system_prompt="You're helpful")
    summary_messages = [m for m in messages if "Summary of earlier conversation" in m["content"]]
    assert len(summary_messages) == 1


