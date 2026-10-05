"""Deterministic guardrails, applied to every message before any LLM call.

The policies name four cases the assistant must not handle: legal topics,
complaints about how an employee treated the customer, billing disputes
(Doc 5) and refunds above $500 (Doc 4). These are fixed business rules, so
they are enforced in code rather than left to the model's judgement. The
system prompt repeats them as a second layer for phrasings the patterns miss.

Matching is keyword-based on lowercased, accent-stripped text, in Spanish and
English. It is biased towards escalating: a needless hand-off to a human is
cheaper than the assistant handling a case it should not.
"""

import re
import unicodedata
from dataclasses import dataclass

SUPPORT_EMAIL = "soporte@tiendahogar.example"
REFUND_APPROVAL_LIMIT = 500.0

LEGAL = "legal"
EMPLOYEE_TREATMENT = "employee_treatment"
BILLING_DISPUTE = "billing_dispute"
REFUND_OVER_LIMIT = "refund_over_limit"


@dataclass(frozen=True)
class GuardrailResult:
    category: str
    reply: str


# --- Patterns (matched against normalized text: lowercase, no accents) -------

_LEGAL = re.compile(
    r"\b(?:abogad[oa]s?|demand(?:a|as|o|e|en|ando|ar\w*)|denunci\w+|i?legal(?:es|mente)?"
    r"|juicio|tribunal(?:es)?|juzgado|litigio|judicial(?:es|mente)?"
    r"|(?:proteccion|defensa|derechos|autoridad(?:es)?) (?:al|del|de) consumidor"
    r"|lawyers?|attorneys?|sue|sued|suing|lawsuits?|court|litigation"
    r"|consumer (?:protection|rights))\b"
)

_STAFF = re.compile(
    r"\b(?:emplead[oa]s?|vendedor(?:a|es|as)?|cajer[oa]s?|repartidor(?:a|es|as)?"
    r"|asesor(?:a|es|as)?|dependient[ea]s?|trabajador(?:a|es|as)?|tecnic[oa]s?|personal"
    r"|gerente|encargad[oa]s?|agentes?|muchach[oa]s?|senorita|joven|(?:me|nos) atendi\w+"
    r"|staff|employees?|salespe(?:rson|ople)|salesm[ae]n|sales ?rep\w*|cashiers?|clerks?"
    r"|representatives?|drivers?|technicians?|manager|guy|lady|worker|associate)\b"
)
_MISTREATMENT = re.compile(
    r"\b(?:grit\w+|insult\w+|groser\w+|maltrat\w+|mal ?trat\w+|irrespet\w+"
    r"|falt\w+ (?:el |al |de )?respeto|humill\w+|amenaz\w+|discrimin\w+|prepotent\w*"
    r"|trat\w+ (?:muy |super |bastante )?(?:mal|pesimo|horrible|fatal|feo)"
    r"|(?:mala|pesima|horrible) atencion|mala actitud|mal ?educad\w+|descort\w+|burl\w+"
    r"|de (?:muy )?mala (?:manera|gana|forma)"
    r"|rude(?:ly|ness)?|yell\w*|shout\w*|scream\w*|disrespect\w*|mistreat\w*|harass\w*"
    r"|humiliat\w*|threaten\w*|mocked|mocking|abusive|unprofessional|impolite)\b"
)
_COMPLAINT = re.compile(r"\b(?:quej\w+|complain\w*)\b")
_TREATMENT = re.compile(r"\b(?:trato|treatment|treated)\b")
# Mistreatment aimed at the customer, with no staff word needed ("me gritaron").
_MISTREATED_ME = re.compile(
    r"\b(?:me|nos) (?:grit|insult|maltrat|humill|amenaz)\w+"
    r"|\bfalt\w+ (?:el |al )?respeto\b"
    r"|\bse burl\w+ de (?:mi|nosotros)\b"
    r"|\b(?:discrimin\w+|acoso|acosad[oa]s?|acosaron|harass\w+)\b"
    r"|\b(?:me|nos) trat\w+ (?:muy |super |bastante )?(?:mal|pesimo|horrible|fatal|feo)\b"
    r"|\b(?:rude|yelled|shouted|screamed|disrespectful|abusive) (?:to|at|with|towards?) (?:me|us)\b"
    r"|\btreated (?:me|us) (?:badly|poorly|terribly|horribly|rudely|like)\b"
)

# A billing word alone ("necesito mi factura") is not a dispute; it needs a cue.
_BILLING_WORD = re.compile(
    r"\b(?:factur\w+|cobr\w+|cargos?|estado de cuenta"
    r"|charg(?:e|es|ed|ing)|bill(?:ed|ing|s)?|invoices?|statement)\b"
)
_DISPUTE_CUE = re.compile(
    r"\b(?:disput\w+|incorrect\w+|equivocad\w+|erro(?:r|res|neo|nea|neos|neas)|indebid\w+"
    r"|dobles?|duplicad\w+|dos veces|de mas|no reconozco|no reconocid\w+|no autori\w+"
    r"|sin autoriza\w+|que (?:yo )?no (?:hice|realice|autorice|compre|pedi|ordene)"
    r"|no (?:coincide|corresponde)"
    r"|wrong(?:ly)?|twice|double|duplicated?|unauthori[sz]ed|not authori[sz]ed?|mistake\w*"
    r"|errors?|unrecogni[sz]ed|unknown"
    r"|(?:don'?t|do not|didn'?t|did not|never)"
    r" (?:recogni[sz]e|make|made|authori[sz]ed?|order(?:ed)?|buy|bought)"
    r"|(?:doesn'?t|does not) match)\b"
)
_BILLING_DISPUTE = re.compile(
    r"\b(?:overcharg\w+|chargebacks?|contracargos?|cobr\w+ (?:mal|demasiado))\b"
)

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
    written in words ("quinientos") are not parsed.
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


def _is_employee_complaint(text: str) -> bool:
    staff = _STAFF.search(text)
    if staff and _MISTREATMENT.search(text):
        return True
    if _COMPLAINT.search(text) and (staff or _TREATMENT.search(text)):
        return True
    return bool(_MISTREATED_ME.search(text))


def _is_billing_dispute(text: str) -> bool:
    if _BILLING_DISPUTE.search(text):
        return True
    return bool(_BILLING_WORD.search(text) and _DISPUTE_CUE.search(text))


# --- Replies ------------------------------------------------------------------

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


_REPLIES = {
    LEGAL: {
        "es": "No puedo ayudarte con temas legales. Este tipo de caso lo atiende un agente "
        f"humano: por favor escribe a {SUPPORT_EMAIL}.",
        "en": "I can't help with legal matters. This kind of case is handled by a human "
        f"agent: please write to {SUPPORT_EMAIL}.",
    },
    EMPLOYEE_TREATMENT: {
        "es": "Lamento lo ocurrido. Las quejas sobre el trato de un empleado las atiende un "
        f"agente humano: por favor escribe a {SUPPORT_EMAIL}.",
        "en": "I'm sorry about what happened. Complaints about how an employee treated you "
        f"are handled by a human agent: please write to {SUPPORT_EMAIL}.",
    },
    BILLING_DISPUTE: {
        "es": "No puedo resolver disputas de facturación. Este tipo de caso lo atiende un "
        f"agente humano: por favor escribe a {SUPPORT_EMAIL}.",
        "en": "I can't resolve billing disputes. This kind of case is handled by a human "
        f"agent: please write to {SUPPORT_EMAIL}.",
    },
    REFUND_OVER_LIMIT: {
        "es": "Los reembolsos mayores a $500 requieren la aprobación de un supervisor "
        "humano, por lo que no puedo aprobar ni procesar esta solicitud.",
        "en": "Refunds over $500 require approval from a human supervisor, so I can't "
        "approve or process this request.",
    },
}


# --- Public API ---------------------------------------------------------------


def check_guardrails(message: str, awaiting_refund_amount: bool = False) -> GuardrailResult | None:
    """Return the escalation for this message, or None if the agent may answer.

    awaiting_refund_amount: the previous customer message asked for a refund
    without an amount, so a bare number in this one ("800") is that amount.
    """
    text = _normalize(message)

    if _LEGAL.search(text):
        category = LEGAL
    elif _is_employee_complaint(text):
        category = EMPLOYEE_TREATMENT
    elif _is_billing_dispute(text):
        category = BILLING_DISPUTE
    else:
        amount = _largest_amount(text)
        about_refund = awaiting_refund_amount or _REFUND.search(text)
        if not (about_refund and amount is not None and amount > REFUND_APPROVAL_LIMIT):
            return None
        category = REFUND_OVER_LIMIT

    return GuardrailResult(category=category, reply=_REPLIES[category][detect_language(message)])


def is_refund_without_amount(message: str) -> bool:
    """True when the message mentions a refund but gives no amount to check."""
    text = _normalize(message)
    return bool(_REFUND.search(text)) and _largest_amount(text) is None
