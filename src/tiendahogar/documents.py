"""The TiendaHogar knowledge base.

The five policies in the policies/ folder are the only source of truth for the
agent. Each file holds the title on the first line and then the text exactly as
supplied with the assessment; they must not be edited, translated or "improved"
for retrieval.
"""

from dataclasses import dataclass
from pathlib import Path

POLICIES_DIR = Path(__file__).parent / "policies"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def load_documents(directory: Path = POLICIES_DIR) -> tuple[Document, ...]:
    documents = []
    for path in sorted(directory.glob("*.txt")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n")
        documents.append(Document(doc_id=path.stem, title=title.strip(), text=body.strip()))
    return tuple(documents)
