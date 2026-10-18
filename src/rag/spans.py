"""Step 6.5 (P55) - strategy-independent hit rule: does a card contain the expected location's TEXT?

The golden set describes locations with structure metadata (article, section path, header
markers), which fixed-size chunks do not have -> fixed was unfairly scored. Here a location is
turned into its text once, from the structure cards that is_hit() selects (the reference
segmentation), and ANY card is scored by word overlap with that text:

  hit  <=>  overlap >= min_share_of_hit * |card|   (the card is mostly this location)
        or  overlap >= min_share_of_location * |location|   (the card holds most of a short one)

Overlap is counted in 8-word shingles (word sequences), lower-cased, punctuation ignored.
Validated in 6.5 on all 61 golden locations: every structure card that is_hit() accepts is also
a span hit; 6 extra cards (G017) are near-identical paragraphs repeated in EBA GL 5.2.5 / 5.2.6.
"""

from __future__ import annotations

import re

from rag.golden import is_hit

WORD = re.compile(r"\w+")


def body(chunk: dict) -> str:
    """Chunk text without the structure context header '[... · ...]' (not part of the source)."""
    text = chunk["text"]
    if chunk.get("strategy") == "structure" and text.startswith("[") and "\n" in text:
        return text.split("\n", 1)[1]
    return text


def shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def location_span(expected: dict, structure_chunks: list[dict], n: int) -> set[tuple[str, ...]]:
    """All shingles of the structure cards that the metadata rule assigns to this location."""
    cards = [c for c in structure_chunks if c["work_id"] == expected["work_id"]]
    out: set[tuple[str, ...]] = set()
    for c in cards:
        if is_hit(c, expected):
            out |= shingles(body(c), n)
    return out


def span_hit(card_shingles: set, span: set, rule: dict) -> bool:
    if not span or not card_shingles:
        return False
    overlap = len(card_shingles & span)
    return overlap >= rule["min_share_of_hit"] * len(card_shingles) or overlap >= rule[
        "min_share_of_location"
    ] * len(span)


def span_ranks(hits: list[dict], spans: list[set], rule: dict) -> list[int | None]:
    """Like metrics.location_ranks, but with the text-overlap rule (one span per location)."""
    n = rule["shingle_words"]
    hit_sh = [(h["rank"], shingles(body(h), n)) for h in hits]
    return [next((r for r, sh in hit_sh if span_hit(sh, span, rule)), None) for span in spans]
