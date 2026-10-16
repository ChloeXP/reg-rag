# reg-rag — EU Banking Regulation RAG Assistant

Bilingual (FR/EN) question-answering over public EU banking regulation
(EU AI Act, CRR, EBA guidelines, ACPR publications), running **fully locally**
with open-source tools, and evaluated systematically (RAGAS, DeepEval, MLflow).

> Status: 🚧 Step 0 / 12 — local environment ready.

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
