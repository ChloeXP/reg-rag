"""Data-quality tests on the real corpus (step 2.7). They need data/raw (downloaded) and
data/processed/chunks.parquet (uv run python -m rag.ingest), so they are marked `data`
and skipped when the files are missing (e.g. in CI)."""

import re
from functools import lru_cache

import pyarrow.parquet as pq
import pytest

from rag.config import PROJECT_ROOT, load_config
from rag.parse_pdf import Footnote, footnote_numbering_gaps, parse_pdf

CFG = load_config()
PARQUET = PROJECT_ROOT / CFG["paths"]["processed_dir"] / "chunks.parquet"
RAW = PROJECT_ROOT / CFG["paths"]["raw_dir"]

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not PARQUET.exists(), reason="run `python -m rag.ingest` first"),
]


@lru_cache(maxsize=1)
def rows() -> list[dict]:
    return pq.read_table(PARQUET).to_pylist()


def structure(doc_id: str) -> list[dict]:
    return [r for r in rows() if r["doc_id"] == doc_id and r["strategy"] == "structure"]


# ---- the two checks asked by the framework ----


def test_every_chunk_has_text_and_doc_id():
    assert all(r["text"].strip() for r in rows())
    assert all(r["doc_id"] for r in rows())


def test_chunk_ids_unique_and_well_formed():
    ids = [r["chunk_id"] for r in rows()]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9_]+::(fixed|structure)::\d{4}", i) for i in ids)


# ---- invariants found while building step 2 ----


def test_no_chunk_over_the_token_limit():
    limit = CFG["chunking"]["chunk_size"]
    assert max(r["n_tokens"] for r in rows()) <= limit


def test_xhtml_has_no_pages_pdf_has_pages():  # decision 2.1-A
    for r in rows():
        if r["format"] == "xhtml":
            assert r["page_start"] is None and r["page_end"] is None
        else:
            assert r["page_start"] is not None


@pytest.mark.parametrize(
    ("doc_id", "n_articles"),
    [("ai_act_en", 113), ("ai_act_fr", 113), ("dora_fr", 64), ("crr_en", 521)],
)
def test_every_article_is_covered(doc_id, n_articles):  # official structure (decision #36)
    articles = {int(r["article"]) for r in structure(doc_id) if r["article"]}
    assert articles == set(range(1, n_articles + 1))


@pytest.mark.parametrize(("doc_id", "n_recitals"), [("ai_act_en", 180), ("dora_fr", 106)])
def test_recitals_complete(doc_id, n_recitals):
    nums = {
        int(m.group(1))
        for r in structure(doc_id)
        if (m := re.match(r"\[.*· (?:Recital|Considérant) (\d+)\]", r["text"]))
    }
    assert nums == set(range(1, n_recitals + 1))


def test_ai_act_en_fr_same_structure():  # bilingual invariant
    def arts(doc_id: str) -> set[str]:
        return {r["article"] for r in structure(doc_id) if r["article"]}

    assert arts("ai_act_en") == arts("ai_act_fr")


@pytest.mark.parametrize("pdf", sorted(p.name for p in RAW.glob("*.pdf")))
def test_one_footnote_per_number(pdf):  # P15 / P16: a continuation line is not a new note
    blocks, _ = parse_pdf(RAW / pdf, CFG)
    notes = [Footnote(b.page, b.text) for b in blocks if b.kind == "footnote"]
    n_found, _, _ = footnote_numbering_gaps(notes)
    assert len(notes) == n_found
