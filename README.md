# reg-rag — EU Banking Regulation RAG Assistant

Bilingual (FR/EN) question-answering over public EU banking regulation
(EU AI Act, CRR, EBA guidelines, ACPR publications), running **fully locally**
with open-source tools, and evaluated systematically (RAGAS, DeepEval, MLflow).

> Status: 🚧 Step 3 / 12 — chunks embedded (bge-m3) in Qdrant; dense, BM25 and hybrid search; 47 tests green (+3 documented xfail).

## Stack

| Component       | Tool                              | Runs              |
|-----------------|-----------------------------------|-------------------|
| LLM             | Ollama (qwen2.5:3b dev, 7b eval)  | native on host    |
| Vector store    | Qdrant v1.19.1                    | Docker            |
| Experiment log  | MLflow v3.16.1                    | Docker            |
| Python env      | uv + Python 3.11 (locked)         | host              |

Ollama runs natively (not in Docker) because Docker on macOS cannot access the Apple Silicon GPU (Metal).

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), [Docker](https://www.docker.com/), [Ollama](https://ollama.com/).
Tested on macOS (Apple Silicon M1, 16 GB).

```bash
# 1. Start Ollama and pull the models (~6.6 GB)
ollama serve &            # or: brew services start ollama
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

# 2. Install the locked Python environment
uv sync

# 3. Start Qdrant + MLflow
docker compose up -d

# 4. Smoke tests
curl -s localhost:6333/healthz   # -> healthz check passed
curl -s localhost:5001/health    # -> OK
uv run python -m rag.llm "Bonjour"

# 5. Download the corpus (10 documents, listed in data/sources.yaml)
uv run python -m rag.download
#    The ACPR paper cannot be fetched by script: the command prints its URL and the
#    exact path to save it to; save it by hand, then re-run the command to register it.

# 6. Parse + chunk everything -> data/processed/chunks.parquet, then run the tests
uv run python -m rag.ingest

# 7. Embed (bge-m3, ~18 min the first time on an M1, then cached) and load Qdrant
uv run python -m rag.embed
uv run python -m rag.index            # add --recreate after a schema change
uv run python scripts/try_search.py   # 5 smoke queries x {dense, bm25, hybrid} x 2 strategies

# 8. Tests: unit (no data) + data-quality (corpus) + integration (Qdrant running)
uv run pytest -q
```

- Qdrant dashboard: http://localhost:6333/dashboard
- MLflow UI: http://localhost:5001 (port 5001 because macOS AirPlay uses 5000)

## Configuration

All parameters live in [`config/config.yaml`](config/config.yaml).
Experiments change the config, never the code.

## Why RAG? (no-RAG baseline)

Asked *"Which AI Act article defines high-risk AI systems?"* (answer: Art. 6 + Annex III),
bare local models invent plausible but wrong articles (Art. 9, 4, 3) and one claims the
AI Act is "not yet adopted". See [`notes/no_rag_baseline.md`](notes/no_rag_baseline.md).

## Data processing (step 2)

Raw file → **Blocks** (one parser per format: PDF via PyMuPDF, CRR via its XHTML markup)
→ **Chunks** (two strategies) → `data/processed/chunks.parquet`.

- Cleaning: headers/footers removed by cross-page repetition; footnotes extracted (kept as
  separate blocks, off by default); TOC and EBA "accompanying documents" (cost-benefit
  analysis, consultation feedback) excluded **declaratively** in `data/sources.yaml`.
- Structure: articles, chapters, recitals, annexes and guideline sections recovered from
  PDF typography; checked against the official structure (AI Act 113 articles /
  180 recitals, DORA 64 / 106, CRR 521) and EN = FR on the bilingual AI Act.
- `fixed`: 512-token windows (64 overlap), counted with the bge-m3 tokenizer.
  `structure`: one legal unit per chunk, split at paragraph boundaries, with a context
  header such as `[CRR · Article 92 Own funds requirements]`.
- Known limits: 107 CRR formulas are images (dropped); 8 EBA footnotes missed (97.7 % recall);
  French needs ~27 % more tokens than English for the same AI Act text.
- CRR Annex IV (correlation table: old directive article → CRR article) is excluded via
  `skip_sections` in `data/sources.yaml`: no regulatory content, and it captured
  article-number queries in step 3.

## Embeddings and retrieval (step 3)

- **Embedding**: `BAAI/bge-m3` (multilingual, 1024-d, normalised) via sentence-transformers on
  Apple MPS (benchmarked: 4.1 vs 2.5 chunks/s on CPU, identical vectors; batch size 8–32
  made no difference). Vectors are cached in `data/processed/embeddings/bge-m3.parquet`,
  keyed by `(chunk_id, sha256(text))` with the encoder signature in the file metadata:
  only changed chunks are re-encoded.
- **Qdrant**: one collection per model × strategy (`bge-m3__fixed`, `bge-m3__structure`),
  each point with two named vectors — `dense` (cosine) and `bm25` (sparse) — and all chunk
  metadata as payload. Point id = `uuid5(chunk_id)`; indexing upserts and deletes stale
  points, so re-runs are idempotent and the collection mirrors the parquet.
- **BM25** as sparse vectors: own tokenizer (lowercase, accents stripped, numbers kept,
  small FR+EN stopword list) and tf/length normalisation (k1 = 1.2, b = 0.75);
  IDF computed server-side by Qdrant (`Modifier.IDF`).
- **Hybrid**: hand-written Reciprocal Rank Fusion (k = 60, 20 candidates per retriever).
- `search(query, cfg, top_k, strategy, filters, mode)` in `src/rag/retrieve.py`;
  `mode` = `dense` (default) | `bm25` | `hybrid`.

Smoke test (5 queries written *before* looking at results, `eval/smoke_queries.yaml`),
rank of the first correct chunk in the top 5 (`structure` strategy):

| Query | Type | dense | bm25 | hybrid |
|---|---|---|---|---|
| Q1 DORA Art. 19 | FR → FR | 1 | 4 | 1 |
| Q2 EBA outsourcing register | EN → EN | 1 | 1 | 1 |
| Q3 CRR Art. 92 | FR → EN (cross-lingual) | 1 | – | 3 |
| Q4 "CRR Article 395" | article number only | 5 | 3 | – |
| Q5 AI Act high-risk (Art. 6 / Annex III) | no-RAG baseline question | 1 | – | 5 |

Takeaways: dense handles paraphrase and cross-lingual queries; BM25 handles rare exact
tokens (with `fixed` chunks it ranks Art. 395 first) but cannot cross languages;
**naive RRF is not a free win** — a chunk found by only one retriever is diluted, and an
article split into several chunks splits its votes. Five hand-picked edge cases are not an
evaluation: weighted RRF, article-level fusion and article-number filtering will be compared
on the golden set in step 6. Default mode therefore stays `dense`.
