"""Step 5 - golden set: blueprint (spec), seed sampling, validation.

Run: uv run python -m rag.golden --check-spec
     uv run python -m rag.golden --sample-seeds      # writes eval/candidates/seeds.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from rag.config import PROJECT_ROOT, load_config

ANSWERABLE = ("factual", "multi_hop", "article_lookup")
BEHAVIORS = {"answer", "refuse", "resist"}


def is_hit(hit: dict, exp: dict) -> bool:
    """Same work AND (article listed OR section starts with a prefix OR text has a marker)."""
    if hit["work_id"] != exp["work_id"]:
        return False
    if hit["article"] in exp.get("articles", []):
        return True
    if any((hit["section"] or "").startswith(s) for s in exp.get("sections", [])):
        return True
    return any(m in hit["text"] for m in exp.get("markers", []))


def load_spec(cfg: dict) -> dict:
    with open(PROJECT_ROOT / cfg["eval"]["golden_spec"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def corpus_profile(cfg: dict, strategy: str = "structure") -> dict[str, dict]:
    """work_id -> {'languages': set, 'kinds': Counter, 'n': int} for one chunking strategy."""
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    rows = pq.read_table(path, columns=["work_id", "language", "kind", "strategy"]).to_pylist()
    prof: dict[str, dict] = {}
    for r in rows:
        if r["strategy"] != strategy:
            continue
        p = prof.setdefault(r["work_id"], {"languages": set(), "kinds": Counter(), "n": 0})
        p["languages"].add(r["language"])
        p["kinds"][r["kind"]] += 1
        p["n"] += 1
    return prof


def check_spec(spec: dict, profile: dict[str, dict]) -> list[str]:
    """Return a list of problems (empty = spec is consistent with itself and with the corpus)."""
    errors = []
    types = spec["types"]
    for split, n in spec["splits"].items():
        got = sum(t[split] for t in types.values())
        if got != n:
            errors.append(f"split {split}: types sum to {got}, expected {n}")
    if sum(spec["splits"].values()) != spec["total"]:
        errors.append("splits do not sum to total")
    if sum(spec["languages"].values()) != spec["total"]:
        errors.append("languages do not sum to total")
    for name, t in types.items():
        if t["behavior"] not in BEHAVIORS:
            errors.append(f"type {name}: unknown behavior {t['behavior']!r}")

    n_answerable = sum(types[t]["dev"] + types[t]["test"] for t in ANSWERABLE)
    n_works = sum(w["n"] for w in spec["works"].values())
    if n_works != n_answerable:
        errors.append(f"works sum to {n_works}, answerable types sum to {n_answerable}")
    for work, w in spec["works"].items():
        if work not in profile:
            errors.append(f"work {work}: not in chunks.parquet")
            continue
        for kind, k in w.get("min_kind", {}).items():
            have = profile[work]["kinds"][kind]
            if have < k:
                errors.append(f"work {work}: needs {k} {kind} chunk(s), corpus has {have}")
    missing = set(profile) - set(spec["works"])
    if missing:
        errors.append(f"works in corpus but not in spec: {sorted(missing)}")

    # Cross-lingual questions are only possible on monolingual works.
    mono_slots = sum(
        w["n"]
        for k, w in spec["works"].items()
        if len(profile.get(k, {}).get("languages", ())) == 1
    )
    if mono_slots < spec["min_cross_lingual"]:
        errors.append(f"only {mono_slots} slots on monolingual works < min_cross_lingual")

    with open(PROJECT_ROOT / spec["smoke_seeds"]["file"], encoding="utf-8") as f:
        smoke = yaml.safe_load(f)
    for q in smoke:
        if q["expected"]["work_id"] not in spec["works"]:
            errors.append(f"smoke {q['id']}: work {q['expected']['work_id']} not in spec")
    if len(smoke) > spec["splits"][spec["smoke_seeds"]["split"]]:
        errors.append("more smoke seeds than questions in their split")
    smoke_types = spec["smoke_seeds"]["types"]
    if set(smoke_types) != {q["id"] for q in smoke}:
        errors.append("smoke_seeds.types must list every smoke query id")
    n_factual = types["factual"]["dev"] + types["factual"]["test"]
    if sum(w["factual"] for w in spec["works"].values()) != n_factual:
        errors.append(f"works.factual does not sum to {n_factual}")
    for work, w in spec["works"].items():
        if not 0 <= w["factual"] <= w["n"]:
            errors.append(f"work {work}: factual must be within 0..n")
    return errors


def load_smoke(spec: dict) -> list[dict]:
    with open(PROJECT_ROOT / spec["smoke_seeds"]["file"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(cfg: dict, strategy: str = "structure") -> list[dict]:
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    rows = pq.read_table(path).to_pylist()
    return [r for r in rows if r["strategy"] == strategy]


def location_key(c: dict) -> tuple[str, str]:
    """One seed per article / annex / section: the same article in FR and EN counts once."""
    if c["article"]:
        return c["work_id"], f"art:{c['article']}"
    annex = re.search(r"ANNEXE? ([IVXL]+)\b", c["section"] or "")
    if annex:
        return c["work_id"], f"annex:{annex.group(1)}"
    recital = re.match(r"\[[^\]]*(?:Considérant|Recital) (\d+)\]", c["text"])
    if recital:  # recitals have neither article nor section: number from the context header
        return c["work_id"], f"rec:{recital.group(1)}"
    return c["work_id"], f"sec:{c['section'] or c['chunk_id']}"


def eligible(c: dict, rules: dict, smoke: list[dict]) -> bool:
    if c["n_tokens"] < rules["min_tokens"] or c["kind"] in rules["exclude_kinds"]:
        return False
    if re.search(rules["exclude_headings"], c["text"].split("\n", 1)[0]):
        return False
    if c["kind"] == "body" and (
        not c["section"] or re.match(rules["exclude_sections"], c["section"])
    ):
        return False
    return not any(is_hit(c, q["expected"]) for q in smoke)  # smoke locations are taken


def seed_plan(spec: dict, chunks: list[dict]) -> dict[str, dict[str, int]]:
    """work -> {kind: number of seeds}. Kinds from min_kind first, the rest from the main kind."""
    smoke_factual = Counter(
        q["expected"]["work_id"]
        for q in load_smoke(spec)
        if spec["smoke_seeds"]["types"][q["id"]] == "factual"
    )
    k = spec["sampling"]["oversample"]
    plan = {}
    for work, w in spec["works"].items():
        total = math.ceil((w["factual"] - smoke_factual[work]) * k)
        kinds = {kind: math.ceil(n * k) for kind, n in w.get("min_kind", {}).items()}
        has_articles = any(c["work_id"] == work and c["kind"] == "article" for c in chunks)
        main = "article" if has_articles else "body"
        kinds[main] = total - sum(kinds.values())
        plan[work] = kinds
    return plan


def spread_pick(pool: list[dict], n: int, rng: random.Random, used: set) -> list[dict]:
    """Systematic sampling: n contiguous buckets over document order, one random chunk each."""
    pool = sorted(pool, key=lambda c: (c["doc_id"], c["index"]))
    picks = []
    for i in range(n):
        bucket = pool[i * len(pool) // n : (i + 1) * len(pool) // n]
        rng.shuffle(bucket)
        for c in bucket:
            if location_key(c) not in used:
                used.add(location_key(c))
                picks.append(c)
                break
    return picks


def sample_seeds(spec: dict, chunks: list[dict]) -> list[dict]:
    rules, smoke = spec["sampling"], load_smoke(spec)
    rng = random.Random(spec["seed"])
    pool = [c for c in chunks if eligible(c, rules, smoke)]
    seeds, used = [], set()
    for work, kinds in seed_plan(spec, chunks).items():
        for kind, n in kinds.items():
            cands = [c for c in pool if c["work_id"] == work and c["kind"] == kind]
            langs = sorted({c["language"] for c in cands})
            # bilingual work: alternate languages so both versions get seeds
            for j, lang in enumerate(langs):
                share = n // len(langs) + (1 if j < n % len(langs) else 0)
                sub = [c for c in cands if c["language"] == lang]
                got = spread_pick(sub, share, rng, used)
                if len(got) < share:
                    raise ValueError(f"{work}/{kind}/{lang}: only {len(got)} of {share} seeds")
                seeds += [(work, kind, c) for c in got]
    out = []
    for i, (work, kind, c) in enumerate(seeds, 1):
        out.append(
            {
                "seed_id": f"S{i:02d}",
                "chunk_id": c["chunk_id"],
                "work_id": work,
                "language": c["language"],
                "kind": kind,
                "article": c["article"],
                "section": c["section"],
                "n_tokens": c["n_tokens"],
                "text_sha": hashlib.sha256(c["text"].encode()).hexdigest()[:12],
            }
        )
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def show_spec(spec: dict, profile: dict[str, dict]) -> None:
    print(f"{'type':<16}{'dev':>5}{'test':>6}  behavior")
    for name, t in spec["types"].items():
        print(f"{name:<16}{t['dev']:>5}{t['test']:>6}  {t['behavior']}")
    total = spec["total"]
    print(f"\n{'work':<17}{'n':>3}{'share':>7}{'chunks':>8}{'corpus':>8}  langs  min_kind")
    n_chunks = sum(p["n"] for p in profile.values())
    for work, w in spec["works"].items():
        p = profile.get(work, {"n": 0, "languages": set()})
        print(
            f"{work:<17}{w['n']:>3}{w['n'] / total:>7.0%}{p['n']:>8}{p['n'] / n_chunks:>8.0%}"
            f"  {'+'.join(sorted(p['languages'])):<5}  {w.get('min_kind', '')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-spec", action="store_true", help="validate eval/golden_spec.yaml")
    ap.add_argument("--sample-seeds", action="store_true", help="write the 5.2 seed chunks")
    args = ap.parse_args()
    cfg = load_config()
    if args.check_spec:
        spec, profile = load_spec(cfg), corpus_profile(cfg)
        show_spec(spec, profile)
        errors = check_spec(spec, profile)
        print("\nspec OK" if not errors else "\n".join(["", *errors]))
        raise SystemExit(1 if errors else 0)
    if args.sample_seeds:
        spec, chunks = load_spec(cfg), load_chunks(cfg)
        seeds = sample_seeds(spec, chunks)
        write_jsonl(seeds, PROJECT_ROOT / spec["sampling"]["output"])
        text = {c["chunk_id"]: c["text"] for c in chunks}
        for s in seeds:
            where = f"art {s['article']}" if s["article"] else (s["section"] or "")[:38]
            head = " ".join(text[s["chunk_id"]].split())[:70]
            print(
                f"{s['seed_id']} {s['work_id']:<15} {s['language']} {s['kind']:<7}"
                f" {where:<38} {head}"
            )
        print(f"\n{len(seeds)} seeds -> {spec['sampling']['output']}")
        print(dict(Counter(s["work_id"] for s in seeds)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
