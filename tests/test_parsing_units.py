"""Fast unit tests (step 2.7): no corpus, no network. Each test pins one rule (or one bug
we hit) from step 2, so it cannot silently come back."""

import pytest

from rag.parse_pdf import (
    Line,
    group_footnotes,
    is_label,
    join_lines,
    parse_page_ranges,
)
from rag.parse_xhtml import parse_xhtml
from rag.schema import Block, Chunk, make_chunk_id
from rag.structure import Para, eu_regulation_structure, size_hierarchy_structure


def line(text: str, page: int = 1, size: float = 9.6, italic: bool = False) -> Line:
    return Line(page, 100.0, 110.0, 842.0, text, size, False, italic)


# ---- data contract (2.1) ----


def test_block_rejects_empty_text():
    with pytest.raises(ValueError):
        Block(doc_id="x", order=0, text="   ", kind="body")


def test_chunk_id_is_deterministic_and_checked():
    assert make_chunk_id("crr_en", "structure", 7) == "crr_en::structure::0007"
    with pytest.raises(ValueError):
        Chunk(
            chunk_id="wrong", doc_id="crr_en", work_id="crr", language="en",
            doc_type="regulation", format="xhtml", strategy="structure", index=0,
            text="t", n_tokens=1, kind="article",
        )  # fmt: skip


def test_chunk_pages_both_or_none():
    with pytest.raises(ValueError):
        Chunk(
            chunk_id=make_chunk_id("d", "fixed", 0), doc_id="d", work_id="w", language="en",
            doc_type="report", format="pdf", strategy="fixed", index=0, text="t",
            n_tokens=1, kind="body", page_start=3, page_end=None,
        )  # fmt: skip


# ---- PDF cleaning (2.2 / 2.4) ----


def test_parse_page_ranges():
    assert parse_page_ranges("2, 30-32") == {2, 30, 31, 32}
    assert parse_page_ranges(None) == set()


def test_join_lines_keeps_compound_hyphens():  # P17
    assert join_lines([line("a risk-"), line("based approach")]) == "a risk-based approach"


def test_join_lines_restores_space_after_number():  # P27
    assert join_lines([line("63.Institutions should")]) == "63. Institutions should"


def test_footnote_continuation_starting_with_a_number():  # P15: "21 May 2024." is not note 21
    notes = group_footnotes(
        [
            line("(3) Position of the European Parliament and decision of the Council of"),
            line("21 May 2024."),
            line("(4) OJ C 517, 22.12.2021, p. 56."),
        ]
    )
    assert [n.text[:3] for n in notes] == ["(3)", "(4)"]


def test_article_label_needs_italics():  # P25 / P26
    assert is_label(line("Article 49", italic=True))
    assert not is_label(line("Article 49", italic=False))  # wrapped cross-reference
    assert is_label(line("CHAPTER III", italic=False))  # EN chapters are not italic


# ---- structure (2.4c) ----


def test_eu_regulation_structure():
    paras = [
        Para("(1) A recital.", False, 9.6),
        Para("CHAPTER I GENERAL PROVISIONS", True, 9.6),
        Para("Article 1 Subject matter", True, 9.6),
        Para("1. This Regulation lays down rules.", False, 9.6),
        Para("Done at Brussels, 13 June 2024.", False, 9.6),
        Para("ANNEX I List", True, 9.6),
        Para("Section A. Something", False, 9.6),
    ]
    kinds = [k for k, _, _ in eu_regulation_structure(paras)]
    assert kinds == ["recital", "title", "title", "article", "body", "title", "annex"]
    assert eu_regulation_structure(paras)[3][1] == "1"  # article number


def test_size_hierarchy_structure():
    paras = [
        Para("4. Guidelines", True, 26),
        Para("12 Pre-outsourcing analysis", True, 16),
        Para("12.2 Risk assessment", True, 16),  # same size: replaces "12 ..."
        Para("64. Institutions should assess...", False, 11),
    ]
    assert size_hierarchy_structure(paras)[-1][2] == "4. Guidelines > 12.2 Risk assessment"


# ---- corpus scope for XHTML (3.4 / P33) ----

XHTML = (
    '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
    '<div id="art_1"><p>Institutions shall hold own funds.</p></div>'
    '<div id="anx_IV"><p>Article 1 | Article 2</p></div>'
    "</body></html>"
)


def test_xhtml_skip_sections(tmp_path):
    path = tmp_path / "doc.xhtml"
    path.write_text(XHTML, encoding="utf-8")
    blocks, stats = parse_xhtml(path, {"anx_IV"})
    assert [b.text for b in blocks] == ["Institutions shall hold own funds."]
    assert stats["skipped_sections"] == ["anx_IV"]
    assert len(parse_xhtml(path)[0]) == 2  # nothing skipped by default
    with pytest.raises(ValueError, match="anx_IX"):  # a typo must fail loudly
        parse_xhtml(path, {"anx_IX"})
