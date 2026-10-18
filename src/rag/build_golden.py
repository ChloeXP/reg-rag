"""Step 5.6 - assemble eval/golden_set.jsonl from the three sources and assign dev / test.

Run: uv run python -m rag.build_golden

Sources: eval/smoke_queries.yaml + eval/smoke_golden.yaml (5, step 3), eval/review/
factual_accepted.jsonl (26, steps 5.3-5.4), eval/handwritten.yaml (29, step 5.5).
Every item is re-validated against the corpus (same checks as rag.handwritten), the split is
stratified by type (spec counts; smoke seeds forced to dev) and, within a type, by
cross-lingual / primary work / language (systematic sampling, spec seed for ties), and each item
records whether its language would be decided by the default in rag.generate.detect_language.
"""

from __future__ import annotations

import json
import random
from collections import Counter

import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.generate import detect_language
from rag.golden import load_smoke, load_spec, write_jsonl
from rag.handwritten import all_chunks, check_item, is_cross, load_items

TYPE_ORDER = ["factual", "multi_hop", "article_lookup", "no_answer", "adversarial"]
OPTIONAL = ("absence_patterns", "injection", "injected_passage", "must_not_match", "must_match",
            "must_cite")  # fmt: skip


def from_smoke(spec: dict, cfg: dict) -> list[dict]:
    with open(PROJECT_ROOT / cfg["eval"]["smoke_golden"], encoding="utf-8") as f:
        extra = {x["id"]: x for x in yaml.safe_load(f)}
    out = []
    for q in load_smoke(spec):
        x = extra[q["id"]]
        out.append(
            {
                "src_id": q["id"],
                "type": spec["smoke_seeds"]["types"][q["id"]],
                "origin": "smoke",
                "verification": "human_location",  # location fixed by hand in step 3
                "question": q["query"],
                "question_language": x["question_language"],
                "expected_behavior": "answer",
                "expected": [q["expected"]],
                "source_chunk_ids": x["source_chunk_ids"],
                "ground_truth": " ".join(x["ground_truth"].split()),
                "key_facts": x["key_facts"],
                "notes": f"smoke query {q['id']} ({q['type']}); ground truth by Claude",
            }
        )
    return out


def from_factual(cfg: dict) -> list[dict]:
    path = PROJECT_ROOT / load_spec(cfg)["review"]["accepted"]
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append(
            {
                "src_id": r["seed_id"],
                "type": "factual",
                "origin": "llm_7b+claude_rewrite",
                "verification": r["verification"],
                "question": r["question"],
                "question_language": r["question_language"],
                "expected_behavior": "answer",
                "expected": [r["expected"]],
                "source_chunk_ids": r["source_chunk_ids"],
                "ground_truth": r["ground_truth"],
                "key_facts": r["key_facts"],
                "notes": r.get("notes"),
            }
        )
    return out


def from_handwritten(cfg: dict) -> list[dict]:
    out = []
    for it in load_items(cfg):
        note = it.get("notes") or ""
        rec = {
            "src_id": it["hid"],
            "type": it["type"],
            "origin": "claude_handwritten",
            "verification": "human_line_by_line" if "HIGH-RISK verified" in note else "claude_only",
            "question": it["question"],
            "question_language": it["question_language"],
            "expected_behavior": it["expected_behavior"],
            "expected": it.get("expected") or [],
            "source_chunk_ids": it.get("source_chunk_ids") or [],
            "ground_truth": " ".join((it.get("ground_truth") or "").split()) or None,
            "key_facts": it.get("key_facts") or [],
            "notes": note,
        }
        rec.update({k: it[k] for k in OPTIONAL if k in it})
        out.append(rec)
    return out


def stratum(it: dict) -> tuple:
    work = it["expected"][0]["work_id"] if it["expected"] else "-"
    return (it["cross_lingual"], work, it["question_language"])


def assign_splits(items: list[dict], spec: dict) -> list[str]:
    """Per type: smoke seeds -> dev; the rest sorted by stratum (cross-lingual, primary work,
    language) and every k-th item -> test (systematic sampling), so test mirrors dev."""
    rng, errors = random.Random(spec["seed"]), []
    for typ in TYPE_ORDER:
        group = sorted((it for it in items if it["type"] == typ), key=lambda x: x["src_id"])
        rest = [it for it in group if it["origin"] != "smoke"]
        for it in group:
            it["split"] = "dev"
        tie = {it["src_id"]: rng.random() for it in rest}
        rest.sort(key=lambda it: (stratum(it), tie[it["src_id"]]))
        n_test = spec["types"][typ]["test"]
        if n_test > len(rest):
            errors.append(f"{typ}: {n_test} test slots but only {len(rest)} non-smoke items")
            continue
        for j in range(n_test):
            rest[int((j + 0.5) * len(rest) / n_test)]["split"] = "test"
    return errors


def apply_patches(items: list[dict], patches: dict) -> list[str]:
    """Apply golden-vN patches in place; return errors. Each patch must hit exactly one item and
    its `old` value must equal the current one (a patch never silently applies elsewhere)."""
    errors = []
    by_src = {it["src_id"]: it for it in items}
    for p in patches["patches"]:
        it = by_src.get(p["src_id"])
        if it is None:
            errors.append(f"{p['id']}: unknown src_id {p['src_id']}")
            continue
        if "fact" in p:
            facts = [f for f in it["key_facts"] if f["fact"] == p["fact"]]
            if len(facts) != 1 or facts[0]["pattern"] != p["old"]:
                errors.append(f"{p['id']}: {p['src_id']} fact {p['fact']!r} / old pattern mismatch")
                continue
            facts[0]["pattern"] = p["new"]
        elif "add_sections" in p:
            hop = it["expected"][p["hop"]]
            hop["sections"] = list(dict.fromkeys([*hop.get("sections", []), *p["add_sections"]]))
        else:
            errors.append(f"{p['id']}: unknown patch kind")
            continue
        it.setdefault("patches", []).append(p["id"])
    return errors


def build(cfg: dict) -> int:
    spec = load_spec(cfg)
    items = from_smoke(spec, cfg) + from_factual(cfg) + from_handwritten(cfg)
    patches = yaml.safe_load((PROJECT_ROOT / cfg["eval"]["golden_patches"]).read_text("utf-8"))
    patch_errors = apply_patches(items, patches)
    for it in items:
        it["golden_version"] = patches["version"]
    corpus = all_chunks(cfg)
    by_id = {c["chunk_id"]: c for c in corpus}
    work_langs: dict[str, set] = {}
    for c in corpus:
        work_langs.setdefault(c["work_id"], set()).add(c["language"])
    errors, warnings = list(patch_errors), []
    for it in items:  # same corpus checks as 5.5, for every source (patched values included)
        check_item({**it, "hid": it["src_id"]}, by_id, corpus, errors, warnings)
    dup = [k for k, n in Counter(it["src_id"] for it in items).items() if n > 1]
    if dup:
        errors.append(f"duplicate source ids: {dup}")
    default = cfg["generation"]["default_language"]
    for it in items:  # step 4.3b: is the answer language decided by the default?
        q = it["question"]
        it["lang_by_default"] = detect_language(q, "fr") != detect_language(q, "en")
        if not it["lang_by_default"] and detect_language(q, default) != it["question_language"]:
            warnings.append(f"{it['src_id']}: detect_language says {detect_language(q, default)}")
        it["cross_lingual"] = is_cross(it, work_langs)
    errors += assign_splits(items, spec)

    items.sort(key=lambda x: (x["split"] != "dev", TYPE_ORDER.index(x["type"]), x["src_id"]))
    rows = []
    for n, it in enumerate(items, 1):
        row = {"id": f"G{n:03d}", "split": it.pop("split"), **it}
        rows.append(row)

    # ---- totals against the blueprint ----
    tab = Counter((r["type"], r["split"]) for r in rows)
    print(f"{'type':<16}{'dev':>5}{'test':>6}")
    for t in TYPE_ORDER:
        print(f"{t:<16}{tab[t, 'dev']:>5}{tab[t, 'test']:>6}")
        for s in ("dev", "test"):
            if tab[t, s] != spec["types"][t][s]:
                errors.append(f"{t}/{s}: {tab[t, s]} != spec {spec['types'][t][s]}")
    langs = Counter(r["question_language"] for r in rows)
    if dict(langs) != spec["languages"]:
        errors.append(f"languages {dict(langs)} != spec {spec['languages']}")
    n_cross = sum(r["cross_lingual"] for r in rows)
    if n_cross < spec["min_cross_lingual"]:
        errors.append(f"cross-lingual {n_cross} < {spec['min_cross_lingual']}")
    primary = Counter(
        r["expected"][0]["work_id"] for r in rows if r["expected_behavior"] == "answer"
    )
    for w, v in spec["works"].items():
        if primary[w] != v["n"]:
            warnings.append(f"work {w}: {primary[w]} answerable questions (spec {v['n']})")
    print(f"total {len(rows)}; languages {dict(langs)}; cross-lingual {n_cross}")
    print(
        f"by split x language: {dict(Counter((r['split'], r['question_language']) for r in rows))}"
    )
    print(f"verification: {dict(Counter(r['verification'] for r in rows))}")
    print(f"origin: {dict(Counter(r['origin'] for r in rows))}")
    by_def = [r["id"] for r in rows if r["lang_by_default"]]
    print(f"language decided by the default ({default}): {len(by_def)} {by_def}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        print("\nnot written")
        return 1
    out = PROJECT_ROOT / cfg["eval"]["golden_set"]
    write_jsonl(rows, out)
    print(f"\n{len(rows)} items -> {cfg['eval']['golden_set']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build(load_config()))
