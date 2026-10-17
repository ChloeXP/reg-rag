# No-RAG baseline (2026-10-16, M1 16GB, Ollama 0.40.5, 100% GPU)

Ground truth: Article 6 (classification rules) + Annex III.
Note: AI Act is Regulation (EU) 2024/1689 — a regulation, not a directive.

| Run | Model      | Setting                         | Answer     | Correct |
|-----|------------|---------------------------------|------------|---------|
| 1   | qwen2.5:3b | ollama CLI, default temp, "une phrase" | Article 9 | No |
| 2   | qwen2.5:7b | ollama CLI, default temp, "une phrase" | Article 4 | No |
| 3   | qwen2.5:3b | rag.llm, temp=0, num_ctx=8192   | Article 3 + calls it a "directive" | No |
| 4   | qwen2.5:7b | rag.llm, temp=0, num_ctx=8192   | Article 3(1), "according to the draft" | No |

Takeaways:
- Bare models hallucinate plausible but wrong article numbers -> motivates RAG.
- Answer changes with wording/sampling -> prompts and sampling params must be versioned.
- Speed (eval rate): 3b ~27 tok/s, 7b ~12.6 tok/s; cold start adds ~3-4 s model load.

## With RAG (step 4.6, 2026-10-17) - same question, same settings + retrieval

Baseline RAG: bge-m3 dense, structure chunks, top-5, prompt v1, temp=0, num_ctx=8192
(`uv run python scripts/smoke_ask.py --models qwen2.5:3b qwen2.5:7b`, log in runs/ask_log.jsonl).

| Model      | No RAG (temp=0)              | RAG                                                        | Correct |
|------------|------------------------------|------------------------------------------------------------|---------|
| qwen2.5:3b | Article 3, calls it a "directive" (run 3) | "[1] defines high-risk AI systems." - cites Annex III, never names the article | Partly |
| qwen2.5:7b | Article 3(1), "according to the draft" (run 4) | "Article 6 of the AI Act defines high-risk AI systems [2]." [2] = Art. 6 Classification rules, p. 53-54 | Yes |

Takeaways:
- temp=0 makes the bare model stable, not right: 3b and 7b both settle on Article 3
  (the definitions article; 3(1) defines "AI system", not high-risk) and 7b still
  believes the Act is a draft -> its knowledge predates the final text (OJ 12.7.2024).
- With retrieval 7b answers correctly and every claim is traceable to a page.
- 3b gets the right passage but does not state the answer: format obeyed, question not
  answered (P40) -> the dev model is not good enough for final answers.
- No-RAG prompt: 41 tokens, 6.9 s; RAG prompt: 1962 tokens, 15.5 s with 7b
  (prefill 14.0 s) -> retrieval costs ~2x latency here, and buys correctness + citations.

Full 5-query smoke test (manual check against the cited text): 3b 2/5, 7b 4/5 correct,
no fabricated fact in any answer. Failures: wrong part of a long article retrieved (Q4,
CRR Art. 395 25 % limit not in top-5), transitional article chosen over Art. 92 (Q3, 3b),
source paragraph numbers written as citations [52][54] (Q2, 7b).
