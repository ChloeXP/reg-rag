"""Structure assignment (step 2.4c): give every PDF paragraph its place in the document.

Input: paragraphs in reading order, each = (text, is_heading, font size).
Output: for each paragraph -> (kind, article, section), the same labels the XHTML parser
reads directly from the CRR markup, so both formats produce identical metadata.

Two strategies, chosen by doc_type in sources.yaml:
- "regulation" (EU Official Journal): headings carry explicit labels
  (CHAPTER / SECTION / Article N / ANNEX), recitals are numbered "(N)" before Article 1.
- everything else (EBA guidelines, reports, ACPR paper): numbered headings whose LEVEL is
  given by font size (bigger = higher), so we keep a stack of open headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.schema import Kind

ARTICLE = re.compile(r"Article\s+(\d+|premier)\b")
CHAPTER = re.compile(r"(CHAPTER|CHAPITRE|TITLE|TITRE)\s+[IVXLC]+\b")
SECTION = re.compile(r"(SECTION|Section)\s+[IVXLC\d]+\b")
ANNEX = re.compile(r"ANNEXE?\b")
RECITAL = re.compile(r"\(\d+\)\s")
FINAL_FORMULA = re.compile(r"(Done at|Fait à)\s")  # "Done at Brussels, 13 June 2024."


@dataclass(frozen=True)
class Para:
    text: str
    is_heading: bool
    size: float


Place = tuple[Kind, str | None, str | None]  # (kind, article, section)


def eu_regulation_structure(paras: list[Para]) -> list[Place]:
    part = "preamble"  # preamble -> articles -> final -> annexes
    chapter = section = article = annex = None
    places: list[Place] = []
    for p in paras:
        text = p.text
        if p.is_heading:
            parent = " > ".join(x for x in (chapter, section) if x) or annex
            if m := ARTICLE.match(text):
                part, annex = "articles", None
                article = "1" if m.group(1) == "premier" else m.group(1)
                places.append(("title", article, parent))
                continue
            if CHAPTER.match(text):
                part, chapter, section, article, annex = "articles", text, None, None, None
            elif SECTION.match(text) and part != "annex":
                section, article = text, None
            elif ANNEX.match(text):
                part, annex, chapter, section, article = "annex", text, None, None, None
            elif part == "annex" and annex:
                annex = f"{annex.split(' > ')[0]} > {text}"  # sub-heading inside an annex
            places.append(("title", None, parent))
            continue

        if part in ("articles", "final") and FINAL_FORMULA.match(text):
            part, article = "final", None
        where = " > ".join(x for x in (chapter, section) if x) or None
        if part == "preamble":
            kind: Kind = "recital" if RECITAL.match(text) else "body"
            places.append((kind, None, None))
        elif part == "articles":
            places.append(("article", article, where))
        elif part == "annex":
            places.append(("annex", None, annex))
        else:
            places.append(("body", None, None))
    return places


def size_hierarchy_structure(paras: list[Para]) -> list[Place]:
    """Headings form a tree by font size: a new heading closes every open heading that is
    the same size or smaller, then opens itself. Body text inherits the open path."""
    stack: list[tuple[float, str]] = []
    places: list[Place] = []
    for p in paras:
        if p.is_heading:
            while stack and stack[-1][0] <= p.size + 0.05:
                stack.pop()
            parent = " > ".join(t for _, t in stack) or None
            stack.append((p.size, p.text))
            places.append(("title", None, parent))
        else:
            places.append(("body", None, " > ".join(t for _, t in stack) or None))
    return places


def assign_structure(paras: list[Para], doc_type: str) -> list[Place]:
    if doc_type == "regulation":
        return eu_regulation_structure(paras)
    return size_hierarchy_structure(paras)
