"""Lightweight local BM25 retrieval over synthetic Finance policies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from rank_bm25 import BM25Okapi


POLICY_DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*(?:\.[0-9]+)?")


class PolicyDocument(TypedDict):
    document_name: str
    text: str


class PolicyChunk(TypedDict):
    document_name: str
    text: str


class RetrievalResult(TypedDict):
    document_name: str
    chunk_text: str
    retrieval_score: float


def load_policy_documents(
    documents_dir: str | Path = POLICY_DOCUMENTS_DIR,
) -> list[PolicyDocument]:
    """Load local Markdown policy documents in deterministic name order."""
    directory = Path(documents_dir)
    return [
        {
            "document_name": path.name,
            "text": path.read_text(encoding="utf-8"),
        }
        for path in sorted(directory.glob("*.md"))
        if path.is_file()
    ]


def split_markdown(document_name: str, text: str) -> list[PolicyChunk]:
    """Split Markdown at headings while retaining each heading as context."""
    chunks: list[PolicyChunk] = []
    current_lines: list[str] = []

    def append_current_chunk() -> None:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(
                {"document_name": document_name, "text": chunk_text}
            )

    for line in text.splitlines():
        if line.lstrip().startswith("#") and current_lines:
            append_current_chunk()
            current_lines = [line]
        else:
            current_lines.append(line)

    append_current_chunk()
    return chunks


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class PolicyRetriever:
    """In-memory BM25 index over the local synthetic policy corpus."""

    def __init__(
        self,
        documents_dir: str | Path = POLICY_DOCUMENTS_DIR,
    ) -> None:
        documents = load_policy_documents(documents_dir)
        self.chunks = [
            chunk
            for document in documents
            for chunk in split_markdown(
                document["document_name"], document["text"]
            )
        ]
        if not self.chunks:
            raise ValueError("No Markdown policy chunks were found.")

        self.index = BM25Okapi(
            [_tokenize(chunk["text"]) for chunk in self.chunks]
        )

    def retrieve(self, question: str, *, top_k: int = 3) -> list[RetrievalResult]:
        """Return the highest-scoring policy chunks for a question."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        question_tokens = _tokenize(question)
        if not question_tokens:
            return []

        scores = self.index.get_scores(question_tokens)
        ranked_indexes = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            {
                "document_name": self.chunks[index]["document_name"],
                "chunk_text": self.chunks[index]["text"],
                "retrieval_score": float(scores[index]),
            }
            for index in ranked_indexes
        ]
