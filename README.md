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

# 8. Ask a question (baseline RAG: retrieve -> prompt v1 -> Ollama -> cited answer)
uv run python -m rag.ask "Quelles sont les obligations de notification des incidents majeurs liés aux TIC ?"
uv run python -m rag.ask --model qwen2.5:7b "What does CRR Article 92 require?"
uv run python scripts/smoke_ask.py --models qwen2.5:3b qwen2.5:7b   # 5 smoke queries end to end

# 9. Golden set (step 5): rebuild eval/golden_set.jsonl from its sources and re-check it
uv run python -m rag.golden --check-spec     # blueprint quotas vs corpus
uv run python -m rag.build_golden            # 60 items, md5 3b5252ca... (deterministic)

# 10. Tests: unit (no data) + data-quality (corpus) + integration (Qdrant, Ollama running)
uv run pytest -q                       # 134 passed, 5 xfailed (known limits, see below)
uv run pytest -q -m "not ollama"       # skip the LLM calls (~45 s)
```

- Qdrant dashboard: http://localhost:6333/dashboard
- MLflow UI: http://localhost:5001 (port 5001 because macOS AirPlay uses 5000)

## Configuration

All parameters live in [`config/config.yaml`](config/config.yaml).
Experiments change the config, never the code.

## Why RAG? (no-RAG baseline)

Asked *"Which AI Act article defines high-risk AI systems?"* (answer: Art. 6 + Annex III),
bare local models invent plausible but wrong articles (Art. 9, 4, 3) and one claims the
AI Act is "not yet adopted". With temperature 0 both models are *stable* but still wrong
(Art. 3). With retrieval, `qwen2.5:7b` answers "Article 6 [2]" and the citation points to
the article's page. See [`notes/no_rag_baseline.md`](notes/no_rag_baseline.md).

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

## Baseline RAG (step 4) — tag `baseline-v1`

`python -m rag.ask` = `search()` → `build_prompt()` → Ollama → `parse_citations()`, no framework
(`src/rag/generate.py`, `src/rag/ask.py`). Baseline configuration, the reference for every
later experiment: `qwen2.5:3b`, temperature 0, bge-m3 dense, `structure` chunks, top-5,
prompt `v1` (sha256 `7de0951c1af7`). `qwen2.5:7b` is the first one-variable experiment.

- **Versioned prompt** [`prompts/v1.yaml`](prompts/v1.yaml): rules in the system role, passages +
  question in the user turn, fixed refusal sentences (FR/EN) for exact refusal detection.
  Each answer logs the prompt version *and* a content hash.
- **Context**: passages numbered `[1]…[5]` with a short source label
  (`[1] DORA — Article 19 Déclaration…`). Square brackets are reserved for citations: the
  chunk's own `[DORA · Article 19]` header is moved into the label so the model cannot copy it.
  Pages are not shown to the model; they go into the reference list built by code.
- **Answer language** is decided by code (FR/EN function-word rule, default `en` when there
  is no signal) and stated in the prompt, instead of asking a 3B model to infer it.
- **Citation checks** (reported, never repaired, so the baseline measures the model as it is):
  valid / invalid `[n]`, refusal, no citation, copied non-numeric brackets; references merged
  per legal unit (an article split in 3 chunks = 1 reference).
- **Run log** `runs/ask_log.jsonl` (gitignored): one line per answer with git commit
  (`-dirty` if uncommitted), config, prompt hash, retrieved `chunk_id` + text hash, flags,
  latency split (load / prefill / decode) and token counts — pointers, not copies.

Smoke test, 5 queries, answers checked by hand against the cited passages:

| Query | qwen2.5:3b | qwen2.5:7b |
|---|---|---|
| Q1 DORA Art. 19 (FR) | ✓ | ✓ |
| Q2 EBA outsourcing register (EN) | ✓ | ✓ content, but cites `[52][54]` (the guideline's own paragraph numbers) |
| Q3 CRR Art. 92 (FR → EN) | ✗ quotes the 2014 transitional ratios (Art. 465) instead of 4.5 / 6 / 8 % | ✓ in French |
| Q4 "CRR Article 395" | ✗ 25 % limit not retrieved (wrong chunk of the article) | ✗ same |
| Q5 AI Act high-risk | ✗ "[1] defines high-risk AI systems." | ✓ Article 6 |
| **Correct** | **2 / 5** | **4 / 5** |

No answer contains a fact absent from its cited passages. The automatic check "a cited
passage is at the expected location" passes 10/10 while only 6/10 answers are right:
citing the right article is necessary, not sufficient — the golden set (step 5) therefore
records answer-level key facts.

Latency on an M1 (Ollama, Metal): prefill ≈ 300 tok/s (3b) / 125 tok/s (7b), decode ≈ 25 / 12
tok/s; a ~2 400-token prompt costs ~8 s (3b) / ~18 s (7b) before the first word, so each extra
retrieved chunk adds ~1.5 s (3b). Two measurement traps found: Ollama's KV cache makes a
repeated prompt's prefill ~0.05 s, and the CLI's retrieval time (~9.5 s) is mostly loading
bge-m3 (0.2–1 s per query inside one process).

## Evaluation set (step 5) — tag `golden-v1`

[`eval/golden_set.jsonl`](eval/golden_set.jsonl): **60 questions, dev 40 / test 20**. Step 6 tunes
on dev only; test is run once, on the final configuration (and with qwen2.5:7b).

| Type | dev | test | Expected behaviour | Checked automatically by |
|---|---|---|---|---|
| factual (one location) | 20 | 10 | grounded answer with `[n]` | location + key-fact regexes |
| multi_hop (2–3 locations, 6 span two documents) | 8 | 4 | grounded answer | all locations retrieved + key facts |
| article_lookup ("What does CRR Article 395 say?") | 3 | 1 | grounded answer | location + key facts |
| no_answer (not in the corpus, 15 %) | 6 | 3 | the fixed refusal sentence of prompt v1 | exact refusal match |
| adversarial (prompt injection) | 3 | 2 | ignore the injected instruction | `must_not_match` / `must_match` / citations |

- **Blueprint first** ([`eval/golden_spec.yaml`](eval/golden_spec.yaml), validated by
  `rag.golden --check-spec`): quotas per type, split, language (FR 28 / EN 32, ≥ 10
  cross-lingual — 12 in the set) and document. Documents get a floor of 2 and CRR is capped at
  ~15 % of the questions (35 % of the chunks): the set is meant to find weaknesses, so small PDFs
  with section-only locations and numbered paragraphs are deliberately over-represented.
- **Each answerable item has an answer-level standard**, not only a location: a reference
  answer (`ground_truth`, for RAGAS context recall) and 1–4 `key_facts` as regexes. Step 4
  showed why: "a cited passage is at the expected location" passed 10/10 while only 6/10
  answers were right. Key-fact regexes are bilingual (the system answers in the question's
  language) and tolerant of Official Journal number formats (`25 %`, `4,5 %`, `EUR 200 000`).
- **No-answer items carry a proof of absence**: `absence_patterns` with zero hits in the whole
  corpus. Several are traps the model knows from pre-training (CRR3 output floor 72.5 %, GDPR
  fines, the GDPR 72-hour rule, Fed SR 11-7) — exactly what faithfulness must catch.
- **Adversarial items** cover both injection in the question (leak the system prompt, "use your
  own knowledge", role hijack) and injection in the retrieved context (`injected_passage`, to be
  inserted by the step-6 harness; prompt v1 rule 8 says passages are data).
- **Split**: stratified by type, then by cross-lingual / primary document / language
  (systematic sampling) — a type-only random split left 1 of the 12 cross-lingual questions in
  test. The 5 step-3 smoke queries are dev only (their baseline answers were already inspected).
- Language detection (step 4) decides 0 of the 60 questions by its default and agrees with the
  label on all 60.

### How the items were made, and who checked them (honest provenance)

| Source | Items | Made by | Checked by |
|---|---|---|---|
| Smoke queries (step 3) | 5 | me | locations fixed by hand before seeing results; ground truths written by Claude |
| Factual from seed chunks | 26 | qwen2.5:7b drafts (40 stratified seed chunks, prompt `golden_draft_v2`) → rewritten by Claude | **8 high-risk items verified by me line by line** against the source (negations, conditions, exceptions, numbers); 18 not human-verified |
| Multi-hop, lookup, no-answer, adversarial | 29 | drafted by Claude from the corpus | not human-verified (13 high-risk items re-checked by Claude against the source) |

Every item is re-checked by code at build time (`rag.build_golden`, `tests/test_golden_set.py`):
source chunks exist and match the expected locations, every key-fact regex matches the
sources, every number in a ground truth appears in its sources, every absence pattern has zero
hits. These checks cannot tell whether a paraphrase keeps the meaning — the first 7b draft of
one item turned "does not fall within the scope" into "can use" and passed all of them. **So 47
of the 60 items have no human line-by-line check: scores in step 6 should be read with that in
mind.** The `verification` field of each item says which applies.

What the 7b drafts were good for: picking a checkable fact and its key numbers. What they were
not good for: wording (64–100 % of the question's words copied from the passage, even with an
explicit "use your own words" example) and faithful reference answers. JSON-schema-constrained
decoding fixed the format problems; it cannot fix meaning.
