"""Integration tests for step 3: Qdrant collections + retrieval on the real corpus.

Marked `qdrant`; skipped when Qdrant is not reachable or chunks.parquet is missing (e.g. CI).
The dense tests load bge-m3 once (~10 s). Known misses from step 3 are xfail on purpose:
they document current limits (journal P33/P34) and turn green when step 6 fixes them.
"""

import httpx
import pyarrow.parquet as pq
import pytest
import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.index import collection_name, get_client, point_id
from rag.retrieve import search

CFG = load_config()
# These tests document the FIRST stage (step 3: dense / bm25 / hybrid ranks + xfails), so the
# reranker of baseline-v2 is switched off here; the default path is tested in test_default_reranks.
CFG_DEFAULT = load_config()
CFG["retrieval"]["rerank"]["enabled"] = False
PARQUET = PROJECT_ROOT / CFG["paths"]["processed_dir"] / "chunks.parquet"
QUERIES = {
    q["id"]: q
    for q in yaml.safe_load((PROJECT_ROOT / "eval" / "smoke_queries.yaml").read_text("utf-8"))
}


def _qdrant_up() -> bool:
    try:
        return httpx.get(f"{CFG['vector_store']['url']}/healthz", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.qdrant,
    pytest.mark.skipif(not PARQUET.exists(), reason="run `python -m rag.ingest` first"),
    pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable (docker compose up -d)"),
]


def first_hit(qid: str, mode: str, strategy: str) -> int | None:
    """Rank of the first expected chunk in the top_k results (same rule as try_search)."""
    exp = QUERIES[qid]["expected"]
    for h in search(QUERIES[qid]["query"], CFG, strategy=strategy, mode=mode):
        if h["work_id"] != exp["work_id"]:
            continue
        if (
            h["article"] in exp.get("articles", [])
            or any((h["section"] or "").startswith(s) for s in exp.get("sections", []))
            or any(m in h["text"] for m in exp.get("markers", []))
        ):
            return h["rank"]
    return None


def test_collections_mirror_the_parquet():  # decision #45: sync, not append
    strategies = pq.read_table(PARQUET, columns=["strategy"]).column("strategy").to_pylist()
    client = get_client(CFG)
    for strategy in set(strategies):
        name = collection_name(CFG, strategy)
        assert client.count(name, exact=True).count == strategies.count(strategy)


def test_payload_round_trip():
    point = get_client(CFG).retrieve(
        collection_name(CFG, "structure"), [point_id("crr_en::structure::0278")]
    )[0]
    assert point.payload["article"] == "92"
    assert point.payload["text"].startswith("[CRR · Article 92 Own funds requirements]")


def test_language_filter():
    hits = search("incident", CFG, strategy="structure", mode="bm25", filters={"language": "fr"})
    assert hits and all(h["language"] == "fr" for h in hits)


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q4", "Q5"])
def test_dense_structure_finds_every_smoke_query(qid):
    assert first_hit(qid, "dense", "structure") is not None


def test_bm25_finds_the_article_number():  # the reason BM25 was added (3.5)
    assert first_hit("Q4", "bm25", "fixed") == 1


@pytest.mark.xfail(reason="P34: RRF dilutes chunks found by one retriever only", strict=False)
@pytest.mark.parametrize("qid", ["Q3", "Q4", "Q5"])
def test_hybrid_structure_at_rank_1(qid):
    assert first_hit(qid, "hybrid", "structure") == 1


def test_default_reranks():
    """baseline-v2 default path: first stage (rerank.candidates) -> cross-encoder -> top_k."""
    k = CFG_DEFAULT["retrieval"]["top_k"]
    hits = search(QUERIES["Q4"]["query"], CFG_DEFAULT)
    assert len(hits) == k and [h["rank"] for h in hits] == list(range(1, k + 1))
    assert all(
        1 <= h["first_stage_rank"] <= CFG_DEFAULT["retrieval"]["rerank"]["candidates"] for h in hits
    )
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
