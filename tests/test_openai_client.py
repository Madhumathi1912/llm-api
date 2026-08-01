import pytest
from unittest.mock import MagicMock, patch
from openai import RateLimitError, AuthenticationError

from app.services.openai_client import OpenAIClient


def _fake_rate_limit_error():
    """
    Builds a real RateLimitError instance the way the OpenAI SDK does
    internally, so our mock raises the exact exception type tenacity
    is configured to catch.
    """
    mock_response = MagicMock()
    mock_response.status_code = 429
    return RateLimitError(
        message="Rate limit hit",
        response=mock_response,
        body={"error": {"message": "Rate limit hit"}},
    )


def test_retries_then_succeeds_on_transient_error():
    """
    Simulates: fail, fail, then succeed on the 3rd attempt.
    Confirms create_completion retries and eventually returns
    the successful result — not the earlier failures.
    """
    client = OpenAIClient()

    fake_success = MagicMock()
    fake_success.choices[0].message.content = "Paris"

    with patch.object(
        client._client.chat.completions,
        "create",
        side_effect=[_fake_rate_limit_error(), _fake_rate_limit_error(), fake_success],
    ) as mock_create:
        result = client.create_completion(model="gpt-4o-mini", messages=[])

    assert result is fake_success
    assert mock_create.call_count == 3  # failed twice, succeeded on 3rd


def test_gives_up_after_max_attempts():
    """
    Simulates: fails every single time (persistent rate limiting).
    Confirms it stops after exactly 3 attempts and raises the
    ORIGINAL RateLimitError — not some wrapped tenacity error.
    """
    client = OpenAIClient()

    with patch.object(
        client._client.chat.completions,
        "create",
        side_effect=_fake_rate_limit_error(),  # always fails
    ) as mock_create:
        with pytest.raises(RateLimitError):
            client.create_completion(model="gpt-4o-mini", messages=[])

    assert mock_create.call_count == 3  # stopped after 3 attempts, didn't retry forever


def test_does_not_retry_on_non_transient_error():
    """
    Simulates an AuthenticationError (bad API key) — this should NOT
    be retried at all, since retrying a bad key gives the same
    failure every time. Confirms it fails immediately (1 attempt only).
    """
    client = OpenAIClient()

    mock_response = MagicMock()
    mock_response.status_code = 401
    auth_error = AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )

    with patch.object(
        client._client.chat.completions,
        "create",
        side_effect=auth_error,
    ) as mock_create:
        with pytest.raises(AuthenticationError):
            client.create_completion(model="gpt-4o-mini", messages=[])

    assert mock_create.call_count == 1  # NO retries — failed once and stopped