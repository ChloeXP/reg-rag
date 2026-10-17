"""Step 3.1 - benchmark bge-m3 encoding on CPU vs MPS with real chunks.

Run: uv run python scripts/bench_embed.py   (decides embedding.device / batch_size)
"""

from __future__ import annotations

import platform
import resource
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

from rag.config import PROJECT_ROOT, load_config

N_SAMPLE = 128
BATCH_SIZES = [8, 16, 32]


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KB on Linux
    return r / 1e9 if platform.system() == "Darwin" else r / 1e6


def main() -> None:
    cfg = load_config()
    model_name = cfg["embedding"]["model"]
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    rng = np.random.default_rng(cfg["project"]["seed"])
    sample = [texts[i] for i in rng.choice(len(texts), N_SAMPLE, replace=False)]
    mps_ok = torch.backends.mps.is_available()
    print(f"arch={platform.machine()} torch={torch.__version__} mps={mps_ok}")
    print(f"sample {N_SAMPLE}/{len(texts)} chunks, model {model_name}\n")

    embs = {}
    for device in ["cpu"] + (["mps"] if mps_ok else []):
        t0 = time.perf_counter()
        model = SentenceTransformer(model_name, device=device)
        n_tok = max(len(model.tokenizer(t)["input_ids"]) for t in sample)
        print(
            f"[{device}] load {time.perf_counter() - t0:.1f}s | longest chunk {n_tok} tokens"
            f" | max_seq_length {model.max_seq_length}"
        )
        model.encode(sample[:8], batch_size=8)  # warm-up: first call compiles kernels
        for bs in BATCH_SIZES:
            t0 = time.perf_counter()
            emb = model.encode(sample, batch_size=bs, normalize_embeddings=True)
            rate = N_SAMPLE / (time.perf_counter() - t0)
            print(
                f"  batch {bs:>2}: {rate:5.1f} chunks/s -> all {len(texts)} in "
                f"{len(texts) / rate / 60:4.1f} min | peak RSS {peak_rss_gb():.1f} GB"
            )
        embs[device] = emb
        del model

    norms = np.linalg.norm(embs["cpu"], axis=1)
    print(f"\ndim={embs['cpu'].shape[1]} | norm min/max={norms.min():.4f}/{norms.max():.4f}")
    if "mps" in embs:
        cos = (embs["cpu"] * embs["mps"]).sum(axis=1)  # vectors are unit-length -> dot = cosine
        print(f"cpu vs mps cosine on same text: min={cos.min():.6f}")


if __name__ == "__main__":
    main()
