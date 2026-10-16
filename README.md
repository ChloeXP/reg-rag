# reg-rag — EU Banking Regulation RAG Assistant

Bilingual (FR/EN) question-answering over public EU banking regulation
(EU AI Act, CRR, EBA guidelines, ACPR publications), running **fully locally**
with open-source tools, and evaluated systematically (RAGAS, DeepEval, MLflow).

> Status: 🚧 Step 2 / 12 — corpus parsed and chunked (2 strategies), 30 tests green.

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
uv run pytest -q          # 10 unit tests + 20 data-quality tests
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
