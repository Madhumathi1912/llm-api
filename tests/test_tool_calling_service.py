import json
from unittest.mock import MagicMock

from app.services.tool_calling_service import ToolCallingService


def test_execute_tool_call_dispatches_to_order_service():
    """
    Confirms _execute_tool_call correctly parses the model's requested
    arguments and routes them to the REAL OrderService method —
    without needing an actual OpenAI call to test this dispatch logic.
    """
    fake_order_service = MagicMock()
    fake_order_service.get_order_status.return_value = {"status": "Shipped"}

    service = ToolCallingService(
        openai_client_=MagicMock(),
        order_service_=fake_order_service,
        cost_logger_=MagicMock(),
    )

    fake_tool_call = MagicMock()
    fake_tool_call.function.name = "get_order_status"
    fake_tool_call.function.arguments = '{"order_id": "ORD1001"}'

    result = service.execute_tool_call(fake_tool_call)

    fake_order_service.get_order_status.assert_called_once_with("ORD1001")
    assert json.loads(result) == {"status": "Shipped"}


def test_execute_tool_call_handles_unknown_function_gracefully():
    """
    If the model somehow requests a function not in TOOLS, we should
    return a clear error payload rather than crashing.
    """
    service = ToolCallingService(
        openai_client_=MagicMock(),
        order_service_=MagicMock(),
        cost_logger_=MagicMock(),
    )

    fake_tool_call = MagicMock()
    fake_tool_call.function.name = "some_unknown_function"
    fake_tool_call.function.arguments = '{}'

    result = service.execute_tool_call(fake_tool_call)
    assert "error" in json.loads(result)


def test_ask_skips_tool_path_when_model_answers_directly():
    """
    When the model's first response has NO tool_calls, ask() should
    return that answer directly — no second API call, no tool execution.
    """
    fake_openai_client = MagicMock()

    fake_message = MagicMock()
    fake_message.tool_calls = None
    fake_message.content = "Paris is the capital of France."

    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    fake_completion.usage.prompt_tokens = 10
    fake_completion.usage.completion_tokens = 5
    fake_completion.model = "gpt-4o-mini"

    fake_openai_client.create_completion.return_value = fake_completion

    service = ToolCallingService(
        openai_client_=fake_openai_client,
        order_service_=MagicMock(),
        cost_logger_=MagicMock(),
    )

    result = service.ask("What is the capital of France?")

    assert result["response"] == "Paris is the capital of France."
    assert result["tool_calls_made"] == []
    # Only ONE call to the model should have happened — no tool round-trip
    assert fake_openai_client.create_completion.call_count == 1


def test_ask_executes_tool_and_makes_second_call_when_requested():
    """
    When the model DOES request a tool call, ask() should: run the real
    tool, send its result back, and make a SECOND call to get the final
    answer — confirming the full two-step flow actually happens.
    """
    fake_openai_client = MagicMock()
    fake_order_service = MagicMock()
    fake_order_service.get_order_status.return_value = {"status": "Shipped"}

    # First call: model requests a tool call
    fake_tool_call = MagicMock()
    fake_tool_call.id = "call_123"
    fake_tool_call.function.name = "get_order_status"
    fake_tool_call.function.arguments = '{"order_id": "ORD1001"}'

    first_message = MagicMock()
    first_message.tool_calls = [fake_tool_call]

    first_completion = MagicMock()
    first_completion.choices = [MagicMock(message=first_message)]

    # Second call: model gives the final natural-language answer
    second_message = MagicMock()
    second_message.content = "Your order ORD1001 has shipped."

    second_completion = MagicMock()
    second_completion.choices = [MagicMock(message=second_message)]
    second_completion.usage.prompt_tokens = 20
    second_completion.usage.completion_tokens = 8
    second_completion.model = "gpt-4o-mini"

    fake_openai_client.create_completion.side_effect = [first_completion, second_completion]

    service = ToolCallingService(
        openai_client_=fake_openai_client,
        order_service_=fake_order_service,
        cost_logger_=MagicMock(),
    )

    result = service.ask("What's the status of order ORD1001?")

    assert result["response"] == "Your order ORD1001 has shipped."
    assert result["tool_calls_made"] == ["get_order_status"]
    assert fake_openai_client.create_completion.call_count == 2  # confirms TWO calls happened
    fake_order_service.get_order_status.assert_called_once_with("ORD1001")