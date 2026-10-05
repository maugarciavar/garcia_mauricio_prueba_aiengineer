"""Agent tests. The OpenAI client is replaced by a scripted fake: no API key,
no network and no LLM call are involved."""

import json
from types import SimpleNamespace

import pytest

from tiendahogar.agent import ORDER_STATUS_TOOL, Agent
from tiendahogar.documents import load_documents
from tiendahogar.guardrails import REFUND_OVER_LIMIT
from tiendahogar.prompts import NO_POLICY_MARKER, SUPPORT_EMAIL, SYSTEM_PROMPT
from tiendahogar.retrieval import RetrievedDocument


def text_reply(text):
    return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text)


def tool_request(name="consultar_estado_pedido", arguments='{"order_id": "ORD-1001"}'):
    call = SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id="call_1")
    return SimpleNamespace(output=[call], output_text="")


class FakeClient:
    """Stands in for openai.OpenAI: returns the scripted replies in order."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        return self._replies.pop(0)


class StubRetriever:
    def __init__(self, *doc_ids):
        by_id = {doc.doc_id: doc for doc in load_documents()}
        self._hits = [RetrievedDocument(document=by_id[doc_id], score=0.9) for doc_id in doc_ids]
        self.calls = []

    def search(self, query, context=()):
        self.calls.append((query, list(context)))
        return list(self._hits)


def make_agent(client, *doc_ids, max_tool_rounds=3):
    return Agent(client, "test-model", StubRetriever(*doc_ids), max_tool_rounds=max_tool_rounds)


def tool_outputs(request):
    return [
        json.loads(item["output"])
        for item in request["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def test_tool_schema_exposes_exactly_the_specified_function():
    assert ORDER_STATUS_TOOL["name"] == "consultar_estado_pedido"
    assert list(ORDER_STATUS_TOOL["parameters"]["properties"]) == ["order_id"]
    assert ORDER_STATUS_TOOL["parameters"]["properties"]["order_id"]["type"] == "string"


def test_refund_over_limit_escalates_without_calling_the_llm():
    client = FakeClient()
    response = make_agent(client).respond("Quiero un reembolso de $800")
    assert (response.route, response.guardrail) == ("escalated", REFUND_OVER_LIMIT)
    assert "supervisor humano" in response.text
    assert SUPPORT_EMAIL not in response.text
    assert client.requests == []


@pytest.mark.parametrize(
    "message",
    ["Voy a demandar a la tienda", "El vendedor me gritó", "Me cobraron dos veces"],
)
def test_cases_recognised_by_meaning_are_sent_to_the_llm_with_the_escalation_rules(message):
    client = FakeClient(text_reply(f"Escribe a {SUPPORT_EMAIL}."))
    response = make_agent(client, "doc5_contacto").respond(message)
    assert response.route == "answered"
    instructions = client.requests[0]["instructions"]
    assert SUPPORT_EMAIL in instructions
    assert "Cases you must not handle" in instructions


def test_system_prompt_states_every_escalation_rule():
    for rule in ("employee", "billing disputes", "legal topic", "over $500", SUPPORT_EMAIL):
        assert rule in SYSTEM_PROMPT


def test_policy_question_sends_the_retrieved_policy_to_the_llm():
    client = FakeClient(text_reply("La garantía es de 12 meses."))
    response = make_agent(client, "doc1_garantia").respond("¿Cuánto dura la garantía de una lavadora?")

    assert (response.route, response.text) == ("answered", "La garantía es de 12 meses.")
    assert [hit.document.doc_id for hit in response.retrieved] == ["doc1_garantia"]
    request = client.requests[0]
    assert "garantía de 12 meses desde la fecha de compra" in request["instructions"]
    assert request["tools"] == [ORDER_STATUS_TOOL]
    assert request["model"] == "test-model"


def test_empty_retrieval_tells_the_llm_no_policy_matched():
    client = FakeClient(text_reply("No tengo esa información."))
    response = make_agent(client).respond("¿Venden televisores?")
    assert response.retrieved == []
    assert NO_POLICY_MARKER in client.requests[0]["instructions"]


def test_valid_order_is_looked_up_and_the_result_returned_to_the_llm():
    client = FakeClient(tool_request(), text_reply("Tu refrigeradora está en tránsito."))
    response = make_agent(client).respond("¿Dónde está mi pedido ORD-1001?")

    assert response.route == "answered"
    assert response.tool_calls[0]["output"]["estado"] == "En tránsito"
    assert tool_outputs(client.requests[1]) == [
        {
            "encontrado": True,
            "order_id": "ORD-1001",
            "producto": "Refrigeradora",
            "estado": "En tránsito",
            "entrega_estimada": "3 días hábiles",
        }
    ]


def test_unknown_order_returns_not_found_to_the_llm():
    client = FakeClient(
        tool_request(arguments='{"order_id": "ORD-9999"}'), text_reply("No encontré ese pedido.")
    )
    make_agent(client).respond("¿Dónde está mi pedido ORD-9999?")
    (output,) = tool_outputs(client.requests[1])
    assert output["estado"] == "no encontrado"
    assert "producto" not in output


@pytest.mark.parametrize(
    "bad_call",
    [
        tool_request(arguments="not json"),
        tool_request(arguments='{"id": "ORD-1001"}'),
        tool_request(name="aprobar_reembolso", arguments='{"order_id": "ORD-1001"}'),
    ],
)
def test_malformed_or_unknown_tool_call_returns_an_error_instead_of_crashing(bad_call):
    client = FakeClient(bad_call, text_reply("No pude consultar el pedido."))
    response = make_agent(client).respond("¿Dónde está mi pedido?")
    assert response.route == "answered"
    assert "error" in tool_outputs(client.requests[1])[0]


def test_tool_loop_is_bounded():
    client = FakeClient(*[tool_request() for _ in range(10)])
    response = make_agent(client, max_tool_rounds=2).respond("¿Dónde está mi pedido ORD-1001?")
    assert response.route == "fallback"
    assert len(client.requests) == 3


def test_conversation_history_is_sent_on_the_next_turn():
    client = FakeClient(text_reply("Son 12 meses."), text_reply("Son 6 meses."))
    agent = make_agent(client, "doc1_garantia")
    agent.respond("¿Garantía de una lavadora?")
    agent.respond("¿Y de una licuadora?")
    assert client.requests[1]["input"] == [
        {"role": "user", "content": "¿Garantía de una lavadora?"},
        {"role": "assistant", "content": "Son 12 meses."},
        {"role": "user", "content": "¿Y de una licuadora?"},
    ]


def test_retrieval_receives_the_previous_customer_messages_as_context():
    client = FakeClient(text_reply("uno"), text_reply("dos"), text_reply("tres"), text_reply("cuatro"))
    retriever = StubRetriever("doc1_garantia")
    agent = Agent(client, "test-model", retriever)
    for message in ["primera", "segunda", "tercera", "cuarta"]:
        agent.respond(message)
    assert retriever.calls == [
        ("primera", []),
        ("segunda", ["primera"]),
        ("tercera", ["primera", "segunda"]),
        ("cuarta", ["segunda", "tercera"]),
    ]


def test_refund_amount_given_in_a_follow_up_message_is_escalated():
    client = FakeClient(text_reply("¿De cuánto es el reembolso?"))
    agent = make_agent(client, "doc4_reembolsos")
    assert agent.respond("Quiero un reembolso").route == "answered"

    response = agent.respond("800")
    assert (response.route, response.guardrail) == ("escalated", REFUND_OVER_LIMIT)
    assert len(client.requests) == 1
