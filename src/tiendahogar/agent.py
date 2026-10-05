"""The support agent: guardrail -> retrieval -> LLM with a bounded tool loop.

Deterministic code handles the deterministic rules (escalation, order lookup);
the model is used only to understand the question, decide whether to look up
an order, and write a grounded answer.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from tiendahogar.config import load_settings
from tiendahogar.guardrails import check_guardrails, detect_language, is_refund_without_amount
from tiendahogar.orders import consultar_estado_pedido
from tiendahogar.prompts import build_instructions
from tiendahogar.retrieval import RetrievedDocument, Retriever, build_retriever

ORDER_STATUS_TOOL = {
    "type": "function",
    "name": "consultar_estado_pedido",
    "description": (
        "Look up the status of a TiendaHogar order by its ID. Returns the product, status "
        "and estimated delivery, or a 'no encontrado' result if the ID does not exist."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID exactly as the customer wrote it, e.g. ORD-1001.",
            }
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Previous customer messages given to retrieval so follow-up questions keep their topic.
_CONTEXT_MESSAGES = 2

_TOOL_LOOP_FALLBACK = {
    "es": "No pude completar tu solicitud en este momento. Por favor, inténtalo de nuevo.",
    "en": "I couldn't complete your request right now. Please try again.",
}


@dataclass
class AgentResponse:
    text: str
    route: str  # "escalated", "answered" or "fallback"
    guardrail: str | None = None
    retrieved: list[RetrievedDocument] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    """One instance holds one conversation."""

    def __init__(self, client: Any, model: str, retriever: Retriever, max_tool_rounds: int = 3):
        self._client = client
        self._model = model
        self._retriever = retriever
        self._max_tool_rounds = max_tool_rounds
        self._history: list[dict[str, str]] = []
        self._awaiting_refund_amount = False

    def respond(self, message: str) -> AgentResponse:
        escalation = check_guardrails(message, self._awaiting_refund_amount)
        if escalation is not None:
            self._awaiting_refund_amount = False
            response = AgentResponse(
                text=escalation.reply, route="escalated", guardrail=escalation.category
            )
        else:
            self._awaiting_refund_amount = is_refund_without_amount(message)
            response = self._answer(message)

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": response.text})
        return response

    def _answer(self, message: str) -> AgentResponse:
        previous = [turn["content"] for turn in self._history if turn["role"] == "user"]
        retrieved = self._retriever.search(message, context=previous[-_CONTEXT_MESSAGES:])
        instructions = build_instructions(retrieved)
        items: list[Any] = [*self._history, {"role": "user", "content": message}]
        tool_calls: list[dict[str, Any]] = []

        for _ in range(self._max_tool_rounds + 1):
            result = self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=items,
                tools=[ORDER_STATUS_TOOL],
            )
            calls = [item for item in result.output if item.type == "function_call"]
            if not calls:
                return AgentResponse(
                    text=result.output_text.strip(),
                    route="answered",
                    retrieved=retrieved,
                    tool_calls=tool_calls,
                )
            items.extend(result.output)
            for call in calls:
                output = _run_tool(call.name, call.arguments)
                tool_calls.append({"name": call.name, "arguments": call.arguments, "output": output})
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )

        # The model kept calling the tool instead of answering.
        return AgentResponse(
            text=_TOOL_LOOP_FALLBACK[detect_language(message)],
            route="fallback",
            retrieved=retrieved,
            tool_calls=tool_calls,
        )


def _run_tool(name: str, arguments: str) -> dict:
    """Execute a tool call requested by the model; never raises."""
    if name != ORDER_STATUS_TOOL["name"]:
        return {"error": f"Unknown tool: {name}"}
    try:
        order_id = json.loads(arguments)["order_id"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"error": "Invalid arguments: expected a JSON object with 'order_id'."}
    return consultar_estado_pedido(order_id)


def build_agent() -> Agent:
    """An agent wired to the real OpenAI client and the local retriever."""
    from openai import OpenAI

    settings = load_settings()
    return Agent(
        client=OpenAI(),
        model=settings.openai_model,
        retriever=build_retriever(),
        max_tool_rounds=settings.max_tool_rounds,
    )
