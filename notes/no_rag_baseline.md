# No-RAG baseline (2026-10-16, M1 16GB, Ollama 0.40.5, 100% GPU)

Ground truth: Article 6 (classification rules) + Annex III.
Note: AI Act is Regulation (EU) 2024/1689 — a regulation, not a directive.

| Run | Model      | Setting                         | Answer     | Correct |
|-----|------------|---------------------------------|------------|---------|
| 1   | qwen2.5:3b | ollama CLI, default temp, "une phrase" | Article 9 | No |
| 2   | qwen2.5:7b | ollama CLI, default temp, "une phrase" | Article 4 | No |
| 3   | qwen2.5:3b | rag.llm, temp=0, num_ctx=8192   | Article 3 + calls it a "directive" | No |
| 4   | qwen2.5:7b | rag.llm, temp=0, num_ctx=8192   | TODO       | TODO    |

Takeaways:
- Bare models hallucinate plausible but wrong article numbers -> motivates RAG.
- Answer changes with wording/sampling -> prompts and sampling params must be versioned.
- Speed (eval rate): 3b ~27 tok/s, 7b ~12.6 tok/s; cold start adds ~3-4 s model load.
