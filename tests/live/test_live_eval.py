"""Live evaluation: runs the scenarios against the real model.

Excluded from the default `pytest` run. Needs OPENAI_API_KEY (in .env or the
environment):

    pytest -m live          # run it
    pytest -m live -rA      # also print every reply

Replies come from an LLM, so an occasional failure can be wording rather than
behaviour: read the printed reply before drawing conclusions.
"""

import os
import unicodedata

import pytest
from dotenv import load_dotenv

from live_scenarios import GENERAL, GUARDRAILS, Scenario
from tiendahogar.agent import Agent
from tiendahogar.config import load_settings

pytestmark = pytest.mark.live


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("’", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


@pytest.fixture(scope="session")
def make_agent(retriever):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client, settings = OpenAI(), load_settings()

    def factory() -> Agent:
        return Agent(client, settings.openai_model, retriever, settings.max_tool_rounds)

    return factory


def check(scenario: Scenario, agent: Agent) -> None:
    for earlier in scenario.previous:
        agent.respond(earlier)
    response = agent.respond(scenario.message)
    reply = normalize(response.text)
    print(f"\n[{scenario.id}] route={response.route} guardrail={response.guardrail}")
    print(f"  retrieved={[(hit.document.doc_id, round(hit.score, 2)) for hit in response.retrieved]}")
    print(f"  tools={[call['arguments'] for call in response.tool_calls]}")
    print(f"  earlier: {list(scenario.previous)}")
    print(f"  Q: {scenario.message}\n  A: {response.text}")

    for expected in scenario.all_of:
        assert normalize(expected) in reply, f"missing {expected!r}"
    if scenario.any_of:
        assert any(normalize(option) in reply for option in scenario.any_of), (
            f"none of {scenario.any_of!r} found"
        )
    for forbidden in scenario.none_of:
        assert normalize(forbidden) not in reply, f"contains {forbidden!r}"
    if scenario.tool_order_id:
        called_with = [call["output"].get("order_id", "").upper() for call in response.tool_calls]
        assert scenario.tool_order_id in called_with, f"tool not called with {scenario.tool_order_id}"


@pytest.mark.parametrize("scenario", GENERAL + GUARDRAILS, ids=lambda scenario: scenario.id)
def test_scenario(scenario, make_agent):
    check(scenario, make_agent())
