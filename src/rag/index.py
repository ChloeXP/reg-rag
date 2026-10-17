"""Load chunk vectors + metadata into Qdrant, one collection per (model, strategy) (step 3.3).

Run: uv run python -m rag.index      (needs: docker compose up -d qdrant, rag.embed cache)

Idempotent: point ids are uuid5(chunk_id), so re-running overwrites the same points, and
points whose chunk no longer exists are deleted -> the collection always mirrors the parquet.
"""

from __future__ import annotations

import uuid

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from rag.config import PROJECT_ROOT, load_config
from rag.embed import embed_chunks, model_slug

# Fixed namespace: the same chunk_id always maps to the same Qdrant point id.
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/ChloeXP/reg-rag")
PAYLOAD_INDEXES = ["doc_id", "work_id", "language", "doc_type", "kind", "article"]
DISTANCES = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def collection_name(cfg: dict, strategy: str) -> str:
    """e.g. 'bge-m3__structure' -> experiments compare collections by name."""
    return f"{model_slug(cfg['embedding']['model'])}__{strategy}"


def get_client(cfg: dict) -> QdrantClient:
    return QdrantClient(url=cfg["vector_store"]["url"], timeout=60)


def ensure_collection(client: QdrantClient, name: str, dim: int, distance: str) -> None:
    """Create the collection if missing; refuse to reuse one built with other vector params."""
    if client.collection_exists(name):
        params = client.get_collection(name).config.params.vectors
        if params.size != dim or params.distance != DISTANCES[distance]:
            raise ValueError(
                f"{name}: existing ({params.size}, {params.distance}) != wanted ({dim}, "
                f"{distance}); delete the collection or change the model name"
            )
        return
    client.create_collection(
        name, vectors_config=models.VectorParams(size=dim, distance=DISTANCES[distance])
    )
    for field in PAYLOAD_INDEXES:  # keyword indexes -> fast filters (language, doc, article)
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def sync_collection(
    client: QdrantClient, name: str, rows: list[dict], vecs: np.ndarray, batch: int
) -> dict:
    """Upsert all rows, then delete points that are not in `rows`. Returns counts."""
    ids = [point_id(r["chunk_id"]) for r in rows]
    for start in range(0, len(rows), batch):
        end = start + batch
        client.upsert(
            name,
            points=models.Batch(
                ids=ids[start:end], vectors=vecs[start:end].tolist(), payloads=rows[start:end]
            ),
            wait=True,
        )
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
    cfg = load_config()
    vs = cfg["vector_store"]
    table = pq.read_table(PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet")
    rows = table.to_pylist()  # payload = every chunk column (text included, for generation)
    vecs, stats = embed_chunks([r["chunk_id"] for r in rows], [r["text"] for r in rows], cfg)
    print(f"vectors {vecs.shape} | embed cache {stats}")

    client = get_client(cfg)
    for strategy in sorted({r["strategy"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["strategy"] == strategy]
        name = collection_name(cfg, strategy)
        ensure_collection(client, name, vecs.shape[1], vs["distance"])
        res = sync_collection(client, name, [rows[i] for i in sel], vecs[sel], vs["upsert_batch"])
        count = client.count(name, exact=True).count
        assert count == len(sel), f"{name}: {count} points != {len(sel)} chunks"
        # round-trip check: the stored vector of the first chunk equals the cached one
        probe = client.retrieve(name, [point_id(rows[sel[0]]["chunk_id"])], with_vectors=True)[0]
        cos = float(np.dot(probe.vector, vecs[sel[0]]))
        print(f"{name}: {count} points | {res} | round-trip cos={cos:.6f}")


if __name__ == "__main__":
    main()
