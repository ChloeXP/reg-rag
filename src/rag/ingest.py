"""Ingestion (step 2.7): raw files -> Blocks -> Chunks (both strategies) -> chunks.parquet.

One command rebuilds the whole processed dataset from data/raw + manifest + config:
    uv run python -m rag.ingest
The output holds BOTH strategies (column `strategy`); config `retrieval.strategy` picks the
one used downstream, so experiments switch strategy without re-ingesting.
"""

from __future__ import annotations

import os
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from rag.chunking import fixed_chunks, load_blocks, load_manifest, structure_chunks
from rag.config import PROJECT_ROOT, load_config
from rag.schema import Chunk

# Explicit schema: nullable ints for pages (XHTML has none, decision 2.1-A)
SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("work_id", pa.string()),
        ("language", pa.string()),
        ("doc_type", pa.string()),
        ("format", pa.string()),
        ("strategy", pa.string()),
        ("index", pa.int32()),
        ("text", pa.string()),
        ("n_tokens", pa.int32()),
        ("kind", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("article", pa.string()),
        ("section", pa.string()),
    ]
)
STRATEGIES = {"fixed": fixed_chunks, "structure": structure_chunks}


def build_chunks(cfg: dict) -> list[Chunk]:
    manifest = load_manifest(cfg)
    raw = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    chunks: list[Chunk] = []
    for doc_id, meta in manifest.items():
        t0 = time.perf_counter()
        blocks = load_blocks(raw / meta["filename"], cfg)
        counts = []
        for name, chunker in STRATEGIES.items():
            doc_chunks = chunker(blocks, meta, cfg)
            chunks.extend(doc_chunks)
            counts.append(f"{name}={len(doc_chunks)}")
        print(
            f"  {doc_id:30} blocks={len(blocks):>5}  {'  '.join(counts)}  "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    return chunks


def write_parquet(chunks: list[Chunk], cfg: dict) -> str:
    """Atomic write: tmp file then rename, so a crash never leaves a half-written dataset."""
    out = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = [c.chunk_id for c in chunks]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate chunk_id: {dupes[:5]}")
    table = pa.Table.from_pylist([c.to_dict() for c in chunks], schema=SCHEMA)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    return str(out.relative_to(PROJECT_ROOT))


def main() -> None:
    cfg = load_config()
    print("Parsing and chunking:")
    chunks = build_chunks(cfg)
    path = write_parquet(chunks, cfg)
    by_strategy = Counter(c.strategy for c in chunks)
    tokens = Counter()
    for c in chunks:
        tokens[c.strategy] += c.n_tokens
    print(f"\nWrote {len(chunks)} chunks to {path}")
    for name, n in by_strategy.items():
        print(f"  {name:9} {n:>5} chunks, {tokens[name]:>8} tokens, mean {tokens[name] // n}")


if __name__ == "__main__":
    main()
