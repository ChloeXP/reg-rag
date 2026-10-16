"""Data contracts for step 2: what parsers produce (Block) and what chunkers produce (Chunk).

Pipeline: raw file --parser (per format)--> list[Block] --chunker (per strategy)--> list[Chunk]
Every parser and every chunker must respect these two shapes, whatever the source format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

# Where a piece of text sits in a legal document.
Kind = Literal["title", "recital", "article", "annex", "body", "footnote", "table"]


@dataclass(frozen=True)
class Block:
    """One structural unit of a document as found by a parser (a paragraph, an article...)."""

    doc_id: str
    order: int  # position in the document, 0-based, strictly increasing
    text: str
    kind: Kind
    page: int | None = None  # 1-based PDF page; None for formats without pages (XHTML)
    article: str | None = None  # e.g. "6", "1" for "Article premier"; None outside articles
    section: str | None = None  # heading path, e.g. "Chapter III > Section 2" or "Annex III"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"{self.doc_id}#{self.order}: empty block text")
        if self.page is not None and self.page < 1:
            raise ValueError(f"{self.doc_id}#{self.order}: page must be >= 1, got {self.page}")


@dataclass(frozen=True)
class Chunk:
    """One retrievable card: text + everything needed to cite and filter it."""

    chunk_id: str  # deterministic: "{doc_id}::{strategy}::{index:04d}"
    doc_id: str
    work_id: str  # same legal text across languages (from manifest)
    language: str  # "fr" | "en"
    doc_type: str  # regulation | guideline | report | discussion_paper
    format: str  # pdf | xhtml
    strategy: str  # chunking strategy that produced it: "fixed" | "structure"
    index: int  # position of the chunk within (doc_id, strategy)
    text: str
    n_tokens: int
    kind: Kind
    page_start: int | None = None  # None for XHTML
    page_end: int | None = None
    article: str | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"{self.chunk_id}: empty chunk text")
        if self.chunk_id != make_chunk_id(self.doc_id, self.strategy, self.index):
            raise ValueError(f"{self.chunk_id}: does not match doc_id/strategy/index")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(f"{self.chunk_id}: page_start and page_end must both be set or None")
        if self.page_start is not None and self.page_end < self.page_start:
            raise ValueError(f"{self.chunk_id}: page_end < page_start")

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk_id(doc_id: str, strategy: str, index: int) -> str:
    """Same input -> same id, so re-running the pipeline overwrites instead of duplicating."""
    return f"{doc_id}::{strategy}::{index:04d}"
