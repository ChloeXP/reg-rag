"""Step 5.3 - draft one golden-set candidate per seed chunk with the local 7b (resumable).

Run: uv run python scripts/gen_candidates.py --limit 2      # try two seeds first
     uv run python scripts/gen_candidates.py                # all remaining seeds
     uv run python scripts/gen_candidates.py --retry-errors # redo failed drafts only

Each draft is appended to eval/candidates/llm_candidates.jsonl as soon as it is written, so
an interrupted run continues where it stopped (per prompt version: a new version
redrafts every seed; 5.4 reads the LAST row per seed_id of the current version).
Code checks (objective, no LLM): JSON keys,
evidence copied verbatim from the chunk, each key fact present in the chunk, and how much
of the question's vocabulary is copied from the passage (copy = too easy for retrieval).
Whether the question is good is judged by hand in step 5.4.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from datetime import UTC, datetime

import yaml

from rag.ask import git_commit
from rag.config import PROJECT_ROOT, load_config
from rag.golden import load_chunks, load_spec
from rag.llm import generate

KEYS = ("question_en", "question_fr", "ground_truth", "key_facts", "evidence")
LANG_NAME = {"en": "English", "fr": "French"}


def norm(text: str) -> str:
    """Compare text modulo case, whitespace, apostrophe style and quote marks."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.sub(r"[«»\"]", " ", text).split())


def load_draft_prompt(spec: dict) -> dict:
    with open(PROJECT_ROOT / spec["drafting"]["prompt"], encoding="utf-8") as f:
        prompt = yaml.safe_load(f)
    parts = [prompt["system"], prompt["user"], json.dumps(prompt.get("schema"), sort_keys=True)]
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
    return {**prompt, "sha256": digest[:12]}


def copy_ratio(question: str, passage: str) -> float:
    """Share of the question's content words (>= 5 letters) that also occur in the passage."""
    words = {w for w in re.findall(r"\w{5,}", norm(question))}
    if not words:
        return 0.0
    text = norm(passage)
    return round(sum(w in text for w in words) / len(words), 2)


def check_draft(raw: str, passage: str, language: str) -> tuple[dict | None, dict]:
    try:
        draft = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, {"json_ok": False, "error": f"JSONDecodeError: {e}"}
    missing = [k for k in KEYS if not draft.get(k)]
    facts = draft.get("key_facts") or []
    facts_is_list = isinstance(facts, list)
    facts = facts if facts_is_list else [facts]
    text = norm(passage)
    question = draft.get(f"question_{language}") or ""
    checks = {
        "json_ok": True,
        "missing_keys": missing,
        "evidence_in_chunk": norm(str(draft.get("evidence", ""))) in text,
        "key_facts_is_list": facts_is_list,
        "facts_in_chunk": [norm(str(f)) in text for f in facts],
        "copy_ratio": copy_ratio(question, passage),
    }
    return draft, checks


def done_ids(path, version: str, retry_errors: bool) -> set[str]:
    """Seeds already drafted with THIS prompt version (a new version redrafts everything)."""
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["prompt_version"] == version]
    return {r["seed_id"] for r in rows if not (retry_errors and r["status"] != "ok")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="draft at most N seeds in this run")
    ap.add_argument("--ids", nargs="*", help="only these seed ids (e.g. S01 S10)")
    ap.add_argument("--retry-errors", action="store_true", help="redo seeds whose status != ok")
    args = ap.parse_args()

    cfg = load_config()
    spec = load_spec(cfg)
    d = spec["drafting"]
    llm_cfg = copy.deepcopy(cfg)
    llm_cfg["llm"].update(model=d["model"], temperature=d["temperature"])
    prompt = load_draft_prompt(spec)

    seeds_path = PROJECT_ROOT / spec["sampling"]["output"]
    seeds = [json.loads(x) for x in seeds_path.read_text(encoding="utf-8").splitlines() if x]
    chunks = {c["chunk_id"]: c for c in load_chunks(cfg)}
    out_path = PROJECT_ROOT / d["output"]
    skip = done_ids(out_path, prompt["version"], args.retry_errors)
    todo = [
        s for s in seeds if s["seed_id"] not in skip and (not args.ids or s["seed_id"] in args.ids)
    ]
    todo = todo[: args.limit] if args.limit else todo
    print(
        f"{len(seeds)} seeds, {len(skip)} already drafted with {prompt['version']}, "
        f"{len(todo)} to draft with {d['model']}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    for i, seed in enumerate(todo, 1):
        chunk = chunks[seed["chunk_id"]]
        sha = hashlib.sha256(chunk["text"].encode()).hexdigest()[:12]
        if sha != seed["text_sha"]:  # corpus re-chunked since sampling: the seed is stale
            raise SystemExit(f"{seed['seed_id']}: chunk text changed since sampling, resample")
        language = LANG_NAME[seed["language"]]
        user = prompt["user"].format(passage=chunk["text"], language=language)
        system = prompt["system"].replace("{language}", language)
        out = generate(user, llm_cfg, system=system, fmt=prompt.get("schema") or "json")
        draft, checks = check_draft(out["text"], chunk["text"], seed["language"])
        ok = checks["json_ok"] and not checks["missing_keys"]
        rec = {
            "seed_id": seed["seed_id"],
            "status": "ok" if ok else "error",
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_commit(),
            "model": out["model"],
            "prompt_version": prompt["version"],
            "prompt_sha": prompt["sha256"],
            **{k: seed[k] for k in ("chunk_id", "work_id", "language", "kind", "article")},
            "section": seed["section"],
            "text_sha": sha,
            "draft": draft,
            "raw": None if ok else out["text"],
            "checks": checks,
            **{k: out[k] for k in ("latency_s", "prompt_tokens", "output_tokens")},
        }
        with open(out_path, "a", encoding="utf-8") as f:  # one line per seed -> resumable
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        q = (draft or {}).get(f"question_{seed['language']}", "")[:70]
        flags = "" if ok else " ERROR"
        if ok and not checks["evidence_in_chunk"]:
            flags += " evidence?"
        if ok and not all(checks["facts_in_chunk"]):
            flags += f" facts {sum(checks['facts_in_chunk'])}/{len(checks['facts_in_chunk'])}"
        print(
            f"[{i}/{len(todo)}] {seed['seed_id']} {out['latency_s']:>5}s "
            f"copy={checks.get('copy_ratio', '-')} {q}{flags}"
        )
    if todo:
        mins = (time.perf_counter() - t_run) / 60
        print(f"\n{len(todo)} drafts in {mins:.1f} min -> {d['output']}")


if __name__ == "__main__":
    main()
