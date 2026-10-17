"""Dense embeddings for chunks, with an on-disk cache (step 3.2).

Run: uv run python -m rag.embed   -> data/processed/embeddings/<model>.parquet

bge-m3 needs ~18 min for the full corpus on an M1 (step 3.1 bench), so vectors are cached:
a chunk is re-encoded only when its (chunk_id, text) changes. The structure-chunk context
header is part of `text` and is embedded on purpose (decision 3.2-A).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rag.config import PROJECT_ROOT, load_config

Encoder = Callable[[Sequence[str]], np.ndarray]


def model_slug(model_name: str) -> str:
    """'BAAI/bge-m3' -> 'bge-m3' (used in cache file and collection names)."""
    return model_name.split("/")[-1]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encoder_signature(cfg: dict) -> str:
    """Anything that changes the vectors must invalidate the cache."""
    e = cfg["embedding"]
    return f"{e['model']}|normalize={e['normalize']}"


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer  # heavy import: only when encoding

    return SentenceTransformer(model_name, device=device)


def make_encoder(cfg: dict, show_progress: bool = False) -> Encoder:
    e = cfg["embedding"]
    model = _load_model(e["model"], e["device"])

    def encode(texts: Sequence[str]) -> np.ndarray:
        vecs = model.encode(
            list(texts),
            batch_size=e["batch_size"],
            normalize_embeddings=e["normalize"],
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    return encode


def embed_query(query: str, cfg: dict) -> np.ndarray:
    """bge-m3 dense needs no query instruction prefix (unlike bge-*-v1.5 models)."""
    return make_encoder(cfg)([query])[0]


def cache_path(cfg: dict) -> Path:
    name = f"{model_slug(cfg['embedding']['model'])}.parquet"
    return PROJECT_ROOT / cfg["paths"]["embeddings_dir"] / name


def load_cache(path: Path, signature: str) -> tuple[dict[tuple[str, str], int], np.ndarray]:
    """(chunk_id, text_sha256) -> row index, plus the vector matrix. Empty if absent/stale."""
    if not path.exists():
        return {}, np.empty((0, 0), dtype=np.float32)
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if meta.get(b"encoder") != signature.encode():
        print(f"cache ignored: encoder changed ({meta.get(b'encoder')} -> {signature})")
        return {}, np.empty((0, 0), dtype=np.float32)
    col = table.column("vector").combine_chunks()
    vecs = col.flatten().to_numpy().reshape(len(table), col.type.list_size)
    ids, hashes = table.column("chunk_id").to_pylist(), table.column("text_sha256").to_pylist()
    keys = zip(ids, hashes, strict=True)
    return {k: i for i, k in enumerate(keys)}, vecs


def write_cache(path: Path, ids, hashes, vecs: np.ndarray, signature: str) -> None:
    """Atomic write (tmp + os.replace), same pattern as download/ingest (#18, #42)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = pa.array(vecs.astype(np.float32).ravel())
    table = pa.table(
        {
            "chunk_id": pa.array(ids, pa.string()),
            "text_sha256": pa.array(hashes, pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(flat, vecs.shape[1]),
        }
    ).replace_schema_metadata({"encoder": signature})
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def embed_chunks(
    ids: Sequence[str], texts: Sequence[str], cfg: dict, encoder: Encoder | None = None
) -> tuple[np.ndarray, dict]:
    """Vectors for `texts` in input order; encodes only cache misses, then rewrites the cache."""
    path, sig = cache_path(cfg), encoder_signature(cfg)
    index, cached = load_cache(path, sig)
    hashes = [text_hash(t) for t in texts]
    keys = list(zip(ids, hashes, strict=True))
    missing = [i for i, k in enumerate(keys) if k not in index]

    new = np.empty((0, 0), dtype=np.float32)
    if missing:
        encoder = encoder or make_encoder(cfg, show_progress=True)
        new = encoder([texts[i] for i in missing])
    dim = new.shape[1] if missing else cached.shape[1]

    out = np.empty((len(texts), dim), dtype=np.float32)
    pos = {i: j for j, i in enumerate(missing)}
    for i, k in enumerate(keys):
        out[i] = new[pos[i]] if i in pos else cached[index[k]]

    stale = len(set(index) - set(keys))
    if missing or stale:
        write_cache(path, list(ids), hashes, out, sig)
    return out, {"encoded": len(missing), "reused": len(texts) - len(missing), "dropped": stale}


def main() -> None:
    cfg = load_config()
    chunks = pq.read_table(
        PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet",
        columns=["chunk_id", "text", "strategy"],
    )
    ids, texts = chunks.column("chunk_id").to_pylist(), chunks.column("text").to_pylist()
    t0 = time.perf_counter()
    vecs, stats = embed_chunks(ids, texts, cfg)
    norms = np.linalg.norm(vecs, axis=1)
    print(f"{len(ids)} chunks -> {vecs.shape} in {time.perf_counter() - t0:.0f}s | {stats}")
    print(f"norm min/max = {norms.min():.4f}/{norms.max():.4f} | cache: {cache_path(cfg)}")


if __name__ == "__main__":
    main()
