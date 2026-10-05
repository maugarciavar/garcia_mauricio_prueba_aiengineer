"""Command-line interface.

    python -m tiendahogar.cli "¿Cuánto dura la garantía de una lavadora?"
    python -m tiendahogar.cli            # interactive conversation
    python -m tiendahogar.cli --debug    # also show route, retrieved policies and tool calls
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from tiendahogar.agent import AgentResponse, build_agent


def _print_response(response: AgentResponse, debug: bool) -> None:
    print(response.text)
    if not debug:
        return
    print(f"  [route] {response.route}" + (f" ({response.guardrail})" if response.guardrail else ""))
    for hit in response.retrieved:
        print(f"  [retrieved] {hit.document.doc_id} score={hit.score:.2f}")
    for call in response.tool_calls:
        print(f"  [tool] {call['name']}({call['arguments']}) -> {call['output']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TiendaHogar support agent")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--debug", action="store_true", help="show how each answer was produced")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 1

    from openai import OpenAIError

    agent = build_agent()
    try:
        if args.question:
            _print_response(agent.respond(args.question), args.debug)
            return 0

        print("Asistente de TiendaHogar. Escribe 'salir' para terminar.")
        while True:
            try:
                message = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message.lower() in {"salir", "exit", "quit"}:
                break
            if message:
                _print_response(agent.respond(message), args.debug)
        return 0
    except OpenAIError as error:
        print(f"The OpenAI request failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
