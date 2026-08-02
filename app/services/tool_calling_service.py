import json

from openai import OpenAIError

from app.config import settings
from app.services.openai_client import OpenAIClient, openai_client
from app.services.order_service import OrderService, order_service
from app.services.cost_logger import CostLogger, cost_logger
from app.services.moderation_client import moderation_client

SYSTEM_PROMPT = "You're a helpful customer support assistant."

# The JSON schema describing what the model is ALLOWED to call — this is
# the "menu" of functions available. The model can only ever request
# calls matching this shape; it cannot invent new functions or
# parameters not listed here.
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_order_status',
            'description': "Get the current status, ETA, and item for a customer's order, given an order ID.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'order_id': {
                        'type': 'string',
                        "description": "The order ID, e.g. 'ORD1001'"
                    }
                },
                'required': ['order_id'],
            }
        }
    }
]

class ToolCallingService:
    """
    Orchestrates the two-step function-calling flow: ask the model what
    (if anything) it wants to call, actually run the real function via
    OrderService, send the real result back, then get the model's final
    natural-language answer. Depends on OpenAIClient and OrderService
    through their public interfaces only.
    """
    def __init__(self, openai_client_: OpenAIClient, order_service_: OrderService, cost_logger_: CostLogger):
        self._openai_client = openai_client_
        self._order_service = order_service_
        self._cost_logger = cost_logger_

    def execute_tool_call(self, tool_call) -> str:
        """
        Dispatches a single tool call request to the REAL function it
        refers to, and returns the result as a JSON string (required
        format for a "tool" role message).
        """
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "get_order_status":
            result = self._order_service.get_order_status(arguments["order_id"])
        else:
            # Model requested something not in TOOLS — shouldn't normally
            # happen, but defend against it rather than crashing.
            result = {"error": f"Unknown function '{function_name}'"}
        return json.dumps(result)


    def ask(self, prompt: str) -> dict:
        """
        Full function-calling flow:
          1. Send the user's prompt + available TOOLS to the model.
          2. If the model requests a tool call, actually run it (real
             Python function, real data) and send the result back.
          3. Get the model's final answer, now grounded in that real data.

        If the model doesn't need any tool for this prompt, it just
        answers directly in step 1 — both paths are handled.
        """
        moderation_client.check(prompt)
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        try:
            first_completion = self._openai_client.create_completion(
                model=settings.openai_model,
                messages=messages,
                tools=TOOLS,
                temperature=0.3
            )
        except OpenAIError as exc:
            raise exc

        first_message = first_completion.choices[0].message
        tool_calls_made = []

        if first_message.tool_calls:
            # The model wants to call one or more real functions.
            # We must echo its own tool-call request back into the
            # conversation before providing the results — the API
            # requires this exact structure.
            messages.append(first_message)

            for tool_call in first_message.tool_calls:
                tool_result = self.execute_tool_call(tool_call)
                tool_calls_made.append(tool_call.function.name)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': tool_result
                })

            try:
                final_completion = self._openai_client.create_completion(
                    model=settings.openai_model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=0.3,
                )
            except OpenAIError as exc:
                raise exc
            
            final_answer = final_completion.choices[0].message.content
            usage = final_completion.usage
            model_used = final_completion.model
        else:
            # No tool needed — the model answered directly on the first call.
            final_answer = first_message.content
            usage = first_completion.usage
            model_used = first_completion.model

        self._cost_logger.log_usage(
            endpoint="/chat/tools",
            model=model_used,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached=False,
        )

        return {
            "response": final_answer,
            "model": model_used,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tool_calls_made": tool_calls_made,
        }


tool_calling_service = ToolCallingService(
    openai_client_=openai_client,
    order_service_=order_service,
    cost_logger_=cost_logger,
)
        