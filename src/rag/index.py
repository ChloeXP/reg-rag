"""Load chunks into Qdrant, one collection per (model, strategy) (steps 3.3 + 3.5).

Run: uv run python -m rag.index [--recreate]   (needs: docker compose up -d qdrant, rag.embed)

Each point carries two named vectors:
  "dense" - bge-m3 embedding (cosine)                        -> semantic search
  "bm25"  - sparse BM25 tf vector, IDF applied by Qdrant      -> keyword search (rag.bm25)
Idempotent: point ids are uuid5(chunk_id), so re-running overwrites the same points, and
points whose chunk no longer exists are deleted -> the collection always mirrors the parquet.
"""

from __future__ import annotations

import argparse
import uuid

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from rag import bm25
from rag.config import PROJECT_ROOT, load_config
from rag.embed import embed_chunks, model_slug

# Fixed namespace: the same chunk_id always maps to the same Qdrant point id.
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/ChloeXP/reg-rag")
PAYLOAD_INDEXES = ["doc_id", "work_id", "language", "doc_type", "kind", "article"]
DISTANCES = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT}
DENSE, SPARSE = "dense", "bm25"  # named vectors


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def collection_name(cfg: dict, strategy: str) -> str:
    """e.g. 'bge-m3__structure' -> experiments compare collections by name."""
    return f"{model_slug(cfg['embedding']['model'])}__{strategy}"


def get_client(cfg: dict) -> QdrantClient:
    return QdrantClient(url=cfg["vector_store"]["url"], timeout=60)


def ensure_collection(
    client: QdrantClient, name: str, dim: int, distance: str, recreate: bool = False
) -> None:
    """Create the collection if missing; refuse to reuse one built with another schema."""
    if client.collection_exists(name) and recreate:
        client.delete_collection(name)
    if client.collection_exists(name):
        params = client.get_collection(name).config.params
        dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
        ok = (
            dense is not None
            and dense.size == dim
            and dense.distance == DISTANCES[distance]
            and SPARSE in (params.sparse_vectors or {})
        )
        if not ok:
            raise ValueError(
                f"{name}: existing schema differs from (dense {dim} {distance} + {SPARSE}); "
                "run `python -m rag.index --recreate`"
            )
        return
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=DISTANCES[distance])},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in PAYLOAD_INDEXES:  # keyword indexes -> fast filters (language, doc, article)
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def sync_collection(
    client: QdrantClient,
    name: str,
    rows: list[dict],
    vecs: np.ndarray,
    sparse: list[tuple[list[int], list[float]]],
    batch: int,
) -> dict:
    """Upsert all rows (dense + sparse), then delete points not in `rows`. Returns counts."""
    ids = [point_id(r["chunk_id"]) for r in rows]
    for start in range(0, len(rows), batch):
        points = [
            models.PointStruct(
                id=ids[i],
                vector={
                    DENSE: vecs[i].tolist(),
                    SPARSE: models.SparseVector(indices=sparse[i][0], values=sparse[i][1]),
                },
                payload=rows[i],
            )
            for i in range(start, min(start + batch, len(rows)))
        ]
        client.upsert(name, points=points, wait=True)
    existing, offset = set(), None
    while True:  # scroll ids only (no payload, no vectors)
        points, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        existing.update(str(p.id) for p in points)
        if offset is None:
            break
    stale = sorted(existing - set(ids))
    if stale:
        client.delete(name, points_selector=models.PointIdsList(points=stale), wait=True)
    return {"upserted": len(ids), "deleted_stale": len(stale)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collections")
    args = ap.parse_args()

    cfg = load_config()
    vs, bm = cfg["vector_store"], cfg["retrieval"]["bm25"]
    table = pq.read_table(PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet")
    rows = table.to_pylist()  # payload = every chunk column (text included, for generation)
    vecs, stats = embed_chunks([r["chunk_id"] for r in rows], [r["text"] for r in rows], cfg)
    print(f"vectors {vecs.shape} | embed cache {stats}")

    client = get_client(cfg)
    for strategy in sorted({r["strategy"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["strategy"] == strategy]
        sel_rows = [rows[i] for i in sel]
        sparse, avgdl = bm25.doc_vectors([r["text"] for r in sel_rows], bm["k1"], bm["b"])
        name = collection_name(cfg, strategy)
        ensure_collection(client, name, vecs.shape[1], vs["distance"], args.recreate)
        res = sync_collection(client, name, sel_rows, vecs[sel], sparse, vs["upsert_batch"])
        count = client.count(name, exact=True).count
        assert count == len(sel), f"{name}: {count} points != {len(sel)} chunks"
        # round-trip check: the stored dense vector of the first chunk equals the cached one
        probe = client.retrieve(name, [point_id(sel_rows[0]["chunk_id"])], with_vectors=[DENSE])
        cos = float(np.dot(probe[0].vector[DENSE], vecs[sel[0]]))
        print(f"{name}: {count} points | {res} | round-trip cos={cos:.6f} | bm25 avgdl={avgdl:.0f}")


if __name__ == "__main__":
    main()
