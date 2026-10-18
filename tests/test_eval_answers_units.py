"""Step 6.6b - context injection (decision 6.6b-D): positions, labels, nothing removed."""

import json

from rag.config import PROJECT_ROOT, load_config
from rag.eval_answers import contexts
from rag.generate import format_context

CFG = load_config()
ITEMS = {
    g["id"]: g
    for g in map(json.loads, (PROJECT_ROOT / CFG["eval"]["golden_set"]).open(encoding="utf-8"))
}
HITS = [
    {"rank": k, "chunk_id": f"dora_fr::structure::{k:04d}", "work_id": "dora",
     "text": f"[DORA · Article {k}]\nbody {k}", "page_start": None, "page_end": None}
    for k in range(1, 6)
]  # fmt: skip


def test_plain_items_get_the_retrieved_passages_unchanged():
    assert contexts(ITEMS["G001"], HITS) == [("none", HITS)]
    assert contexts(ITEMS["G038"], HITS) == [("none", HITS)]  # question injection: nothing added


def test_context_injection_goes_first_then_last_and_removes_nothing():
    g = ITEMS["G040"]
    (v1, first), (v2, last) = contexts(g, HITS)
    assert (v1, v2) == ("first", "last")
    assert first[0]["text"] == g["injected_passage"] and first[1:] == HITS
    assert last[-1]["text"] == g["injected_passage"] and last[:-1] == HITS


def test_poisoned_passage_looks_like_a_document():
    _, first = contexts(ITEMS["G040"], HITS)[0]
    ctx = format_context(first)
    assert ctx.startswith("[1] EXTERNAL NOTE\n")
    assert "inject" not in ctx.lower()
