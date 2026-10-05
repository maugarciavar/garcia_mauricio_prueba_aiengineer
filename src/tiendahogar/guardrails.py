"""Deterministic guardrail for the one rule that is arithmetic: refunds over $500.

Doc 4 sets a hard limit: refunds above $500 need a human supervisor. Comparing
an amount with a limit is not a language task, so it is enforced in code, on
every message, before any LLM call. A request that trips it never reaches the
model, which also makes it immune to prompt injection.

The other cases the assistant must not handle (legal topics, complaints about
employee treatment, billing disputes) are recognised by meaning, not by a
number, and are left to the system prompt. An earlier version matched them
with keyword patterns here; in the live evaluation the prompt alone caught
every such case, including paraphrases the patterns missed, so the patterns
were removed. See "Arquitectura propuesta y justificación" in SUBMISSION.md.
"""

import re
import unicodedata
from dataclasses import dataclass

REFUND_APPROVAL_LIMIT = 500.0
REFUND_OVER_LIMIT = "refund_over_limit"


@dataclass(frozen=True)
class GuardrailResult:
    category: str
    reply: str


# --- Patterns (matched against normalized text: lowercase, no accents) -------

_REFUND = re.compile(
    r"\b(?:reembols\w+|reintegr\w+|refund\w*|money back"
    r"|devol\w+ (?:del |de mi |el |mi )dinero|devuelv\w+ (?:el |mi )dinero"
    r"|(?:me|nos) (?:devuelv|regres|reintegr)\w+|(?:devuelv|regres)\w*(?:me|nos))\b"
)

# Numbers that are not money and must not be read as a refund amount.
_ORDER_ID = re.compile(r"\bord[-\s]?\d+\b")
_DURATION = re.compile(
    r"\d+(?:\s*(?:-|a|o|to|or)\s*\d+)?\s*"
    r"(?:dias?|mes(?:es)?|anos?|semanas?|horas?|days?|months?|years?|weeks?|hours?|business)\b"
)
_PERCENTAGE = re.compile(r"\d+(?:[.,]\d+)?\s*%")

_THOUSANDS_SUFFIX = re.compile(r"(\d+)\s*(?:mil|k)\b")
_THOUSAND_WORD = re.compile(
    r"\b(?:mil|(?:a |one )?thousand)(?= (?:dolar|dollar|bucks|pesos|quetzales|usd))"
)
_NUMBER = re.compile(r"(?:(?<![a-z\d.,])|(?<=\bq)|(?<=usd))(?:\d[\d.,]*\d|\d)")
_CURRENCY_BEFORE = re.compile(r"(?:us\$|\$|usd|\bq)\s*$")
_CURRENCY_AFTER = re.compile(r"\s*(?:usd|dolar(?:es)?|dollars?|bucks|pesos|quetzales)\b")


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _parse_number(raw: str) -> float | None:
    """Read "1,200.50", "1.200,50", "1.200" (thousands) or "500,50" (decimal)."""
    if "." in raw and "," in raw:
        decimal = "." if raw.rfind(".") > raw.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        raw = raw.replace(thousands, "").replace(decimal, ".")
    elif re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
        raw = raw.replace(".", "").replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _largest_amount(normalized: str) -> float | None:
    """The largest money amount in the text, or None.

    Amounts carrying a currency marker win; if none has one, every remaining
    number counts. Apart from "mil"/"thousand" before a currency, amounts
    written in words ("quinientos") are not parsed; the system prompt covers them.
    """
    text = _ORDER_ID.sub(" ", normalized)
    text = _DURATION.sub(" ", text)
    text = _PERCENTAGE.sub(" ", text)
    text = _THOUSANDS_SUFFIX.sub(lambda match: str(int(match.group(1)) * 1000), text)
    text = _THOUSAND_WORD.sub("1000", text)

    marked, unmarked = [], []
    for match in _NUMBER.finditer(text):
        value = _parse_number(match.group())
        if value is None:
            continue
        has_currency = _CURRENCY_BEFORE.search(text[: match.start()]) or _CURRENCY_AFTER.match(
            text[match.end() :]
        )
        (marked if has_currency else unmarked).append(value)

    amounts = marked or unmarked
    return max(amounts) if amounts else None


# --- Reply --------------------------------------------------------------------
# Doc 4 requires a human supervisor for these refunds but names no contact channel,
# and Doc 5 assigns the support email to other cases only, so the reply gives none.

_SPANISH_WORDS = frozenset(
    "el la los las de del que mi un una es por para con quiero como cuanto se lo y en al fue "
    "le muy pero porque esta este hola gracias necesito tengo".split()
)
_ENGLISH_WORDS = frozenset(
    "the i my is to and you your of for want was do can how what please with this that it "
    "they have on in am are not hello thanks need get".split()
)


def detect_language(message: str) -> str:
    """"en" when the message looks English, otherwise "es" (the default)."""
    words = re.findall(r"[a-z']+", _normalize(message))
    english = sum(word in _ENGLISH_WORDS for word in words)
    spanish = sum(word in _SPANISH_WORDS for word in words)
    return "en" if english > spanish else "es"


_REFUND_OVER_LIMIT_REPLY = {
    "es": "Los reembolsos mayores a $500 requieren la aprobación de un supervisor "
    "humano, por lo que no puedo aprobar ni procesar esta solicitud.",
    "en": "Refunds over $500 require approval from a human supervisor, so I can't "
    "approve or process this request.",
}


# --- Public API ---------------------------------------------------------------


def check_guardrails(message: str, awaiting_refund_amount: bool = False) -> GuardrailResult | None:
    """Return the escalation for this message, or None if it may go to the model.

    awaiting_refund_amount: the previous customer message asked for a refund
    without an amount, so a bare number in this one ("800") is that amount.
    """
    text = _normalize(message)
    amount = _largest_amount(text)
    about_refund = awaiting_refund_amount or _REFUND.search(text)
    if about_refund and amount is not None and amount > REFUND_APPROVAL_LIMIT:
        return GuardrailResult(
            category=REFUND_OVER_LIMIT, reply=_REFUND_OVER_LIMIT_REPLY[detect_language(message)]
        )
    return None


def is_refund_without_amount(message: str) -> bool:
    """True when the message mentions a refund but gives no amount to check."""
    text = _normalize(message)
    return bool(_REFUND.search(text)) and _largest_amount(text) is None
