"""Step 5.4 - human review of the 7b drafts through a YAML review sheet.

Run: uv run python -m rag.review --make-sheet          # once: writes the review sheet
     uv run python -m rag.review --show CHUNK_ID [-n 1] # print a chunk (+ n neighbours each side)
     uv run python -m rag.review --apply                # check the sheet; write accepted if all OK

The sheet holds, per seed, the source chunk (as YAML comments) and the draft fields to edit.
`--apply` never trusts the reviewer blindly: it re-checks every accepted item against the
source text (regex key facts, copy ratio, expected location), then assigns question
languages to meet the quota and writes eval/review/factual_accepted.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from collections import Counter

import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.golden import is_hit, load_chunks, load_smoke, load_spec, location_key, write_jsonl

YES_NO = re.compile(r"^(can|is|are|does|do|should|must|may|est-ce|peut-on|existe-t-il)\b", re.I)
META = re.compile(
    r"(the passage|this text|the document|ce document|le passage|ce texte|article \d)", re.I
)
WORDS = re.compile(r"\w{5,}")


def norm(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def copy_ratio(question: str, source: str) -> float:
    words = set(WORDS.findall(norm(question)))
    text = norm(source)
    return round(sum(w in text for w in words) / len(words), 2) if words else 0.0


THOUSANDS = re.compile(r"\d{1,3}(?:[ ,.\u00a0\u202f]\d{3})+")


def fact_pattern(fact: str) -> str:
    """Literal fact -> tolerant regex: flexible spaces, '25 %' == '25%', both apostrophes,
    any thousands separator ('EUR 200,000' matches the EU-style 'EUR 200 000') and either
    decimal mark ('4.5 %' matches the Official Journal's '4,5 %')."""
    parts, last = [], 0
    for m in THOUSANDS.finditer(fact.strip()):
        parts.append(re.escape(fact.strip()[last : m.start()]))
        parts.append(r"[\s,.]?".join(re.split(r"[ ,.\u00a0\u202f]", m.group())))
        last = m.end()
    parts.append(re.escape(fact.strip()[last:]))
    pat = re.sub(r"(\d)(?:\\\.|,)(\d)", r"\1[.,]\2", "".join(parts))  # 4.5 == EU-style 4,5
    pat = pat.replace(r"\ ", r"\s+").replace(r"\s+%", r"\W{0,3}%")  # after: {0,3} has a comma
    return re.sub(r"['’]", "['’]", pat)


def latest_drafts(spec: dict) -> dict[str, dict]:
    """Last row per seed_id for the current draft prompt version (retries append rows)."""
    with open(PROJECT_ROOT / spec["drafting"]["prompt"], encoding="utf-8") as f:
        version = yaml.safe_load(f)["version"]
    rows = {}
    with open(PROJECT_ROOT / spec["drafting"]["output"], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["prompt_version"] == version and r["status"] == "ok":
                rows[r["seed_id"]] = r
    return rows


def default_expected(seed: dict, chunks: list[dict]) -> dict:
    """Location in the smoke_queries format, both languages when the work has two."""
    work, kind = seed["work_id"], seed["kind"]
    if seed["article"]:
        heads = {
            c["text"].split("\n", 1)[0].split(" · ", 1)[1].rstrip("]")
            for c in chunks
            if c["work_id"] == work and c["article"] == seed["article"]
        }
        return {"work_id": work, "articles": [seed["article"]], "markers": sorted(heads)}
    key = location_key(next(c for c in chunks if c["chunk_id"] == seed["chunk_id"]))[1]
    if key.startswith("rec:"):
        n = key[4:]
        return {"work_id": work, "markers": [f"Recital {n}]", f"Considérant {n}]"]}
    if key.startswith("annex:"):
        roman = key[6:]
        return {"work_id": work, "sections": [f"ANNEX {roman} ", f"ANNEXE {roman} "]}
    assert kind == "body", seed
    return {"work_id": work, "sections": [seed["section"]]}


def flags(draft: dict, chk: dict, language: str) -> list[str]:
    q = draft[f"question_{language}"] + " " + draft["question_en"]
    out = []
    if not chk["evidence_in_chunk"]:
        out.append("evidence not verbatim")
    if not all(chk["facts_in_chunk"]):
        out.append(f"facts {sum(chk['facts_in_chunk'])}/{len(chk['facts_in_chunk'])} in source")
    if YES_NO.match(draft["question_en"]) or YES_NO.match(draft["question_fr"]):
        out.append("yes/no")
    if META.search(q):
        out.append("mentions passage/document/article")
    return out


def dq(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)  # JSON string == valid YAML double-quoted scalar


def sq(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"  # YAML single-quoted: backslashes stay literal


def block(text: str, indent: int) -> str:
    pad = " " * indent
    lines = textwrap.wrap(" ".join(text.split()), 92 - indent) or [""]
    return "|-\n" + "\n".join(pad + line for line in lines)


HEADER = """# Step 5.4 review sheet - factual questions drafted by {model} (prompt {version}).
# Edit THIS file by hand, then run: uv run python -m rag.review --apply
#
# decision: accept (goes into the golden set, edited or not) | drop (+ drop_reason) | todo
# Accept exactly the number of items per work shown below (spec: factual - smoke seeds).
#
# Checklist per accepted item (the review criteria, also listed in the README):
#  1. Understandable WITHOUT the source: names the regulation and the context.
#  2. One question, not yes/no, about the regulation (not about the document itself).
#  3. User wording, not the legal phrasing (apply warns above copy {max_copy}).
#  4. ground_truth checked sentence by sentence against SOURCE: negations, scope, conditions.
#     Keep it in the SOURCE language (5.3-C); 1-3 sentences.
#  5. key_facts: 1-4 facts a correct answer must contain; `pattern` is a case-insensitive
#     Python regex that must match the source chunk(s). Prefer numbers / dates / bodies.
#  6. Answer needs a neighbouring chunk? add its id to source_chunk_ids
#     (see: uv run python -m rag.review --show CHUNK_ID -n 1).
#  7. question_language: auto (code assigns to meet FR/EN + cross-lingual quotas) or fr / en.
#
# To accept per work: {need}
"""


def make_sheet(cfg: dict, spec: dict, force: bool) -> None:
    path = PROJECT_ROOT / spec["review"]["sheet"]
    if path.exists() and not force:
        raise SystemExit(f"{path} exists: it may hold your edits. Use --force to overwrite.")
    chunks = load_chunks(cfg)
    text = {c["chunk_id"]: c["text"] for c in chunks}
    drafts = latest_drafts(spec)
    seeds = [
        json.loads(x) for x in open(PROJECT_ROOT / spec["sampling"]["output"], encoding="utf-8")
    ]
    first = next(iter(drafts.values()))
    out = [
        HEADER.format(
            model=first["model"],
            version=first["prompt_version"],
            max_copy=spec["review"]["max_copy_ratio"],
            need=needed_per_work(spec),
        )
    ]
    for seed in seeds:
        r = drafts.get(seed["seed_id"])
        if r is None:
            continue
        d, chk = r["draft"], r["checks"]
        flag_text = ", ".join(flags(d, chk, seed["language"])) or "-"
        src = text[seed["chunk_id"]]
        out.append("# " + "=" * 88)
        out.append(
            f"# {seed['seed_id']} · {seed['work_id']} · {seed['language']} · {seed['kind']}"
            f" · copy {chk['copy_ratio']} · flags: {flag_text}"
        )
        out.append("# ---- SOURCE " + seed["chunk_id"] + " (read-only) ----")
        for para in src.split("\n"):
            for line in textwrap.wrap(para, 90) or [""]:
                out.append("#   " + line)
        facts = "\n".join(
            f"    - fact: {dq(f)}\n      pattern: {sq(fact_pattern(f))}" for f in d["key_facts"]
        )
        exp = default_expected(seed, chunks)
        out.append(
            f"- seed_id: {seed['seed_id']}\n"
            f"  decision: todo\n"
            f"  drop_reason: null\n"
            f"  question_language: auto\n"
            f"  question_en: {dq(d['question_en'])}\n"
            f"  question_fr: {dq(d['question_fr'])}\n"
            f"  ground_truth: {block(d['ground_truth'], 4)}\n"
            f"  key_facts:\n{facts}\n"
            f"  source_chunk_ids: [{seed['chunk_id']}]\n"
            f"  expected: {json.dumps(exp, ensure_ascii=False)}\n"
            f"  notes: null\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    # round-trip guard: the sheet must give back exactly the drafted text
    for it in items:
        d = drafts[it["seed_id"]]["draft"]
        assert it["question_en"] == d["question_en"] and it["question_fr"] == d["question_fr"]
        assert [f["fact"] for f in it["key_facts"]] == d["key_facts"]
    print(f"{len(items)} items -> {path.relative_to(PROJECT_ROOT)} (round-trip OK)")
    print(f"to accept per work: {needed_per_work(spec)}")


def needed_per_work(spec: dict) -> dict[str, int]:
    smoke = Counter(
        q["expected"]["work_id"]
        for q in load_smoke(spec)
        if spec["smoke_seeds"]["types"][q["id"]] == "factual"
    )
    return {w: v["factual"] - smoke[w] for w, v in spec["works"].items()}


def check_item(it: dict, draft: dict, chunks: dict, spec: dict) -> tuple[list, list]:
    errors, warnings = [], []
    sid = it["seed_id"]
    note = it.get("notes") or ""
    if "verify against SOURCE" in note or "HIGH-RISK: verify" in note:  # 5.4-B: human verifies
        errors.append(f"{sid}: not verified yet - check it against SOURCE, then edit the note")
    if it.get("question_language") not in ("auto", "fr", "en"):
        errors.append(f"{sid}: question_language must be auto / fr / en")
    for lang in ("en", "fr"):
        if not (it.get(f"question_{lang}") or "").strip():
            errors.append(f"{sid}: question_{lang} is empty")
    gt = (it.get("ground_truth") or "").strip()
    if not gt:
        errors.append(f"{sid}: ground_truth is empty")
    ids = it.get("source_chunk_ids") or []
    missing = [i for i in ids if i not in chunks]
    if not ids or missing:
        errors.append(f"{sid}: source_chunk_ids empty or unknown: {missing}")
        return errors, warnings
    source = "\n".join(chunks[i]["text"] for i in ids)
    facts = it.get("key_facts") or []
    if not 1 <= len(facts) <= 4:
        errors.append(f"{sid}: needs 1-4 key_facts, has {len(facts)}")
    for f in facts:
        try:
            if not re.search(f["pattern"], source, re.I):
                errors.append(f"{sid}: pattern {f['pattern']!r} does not match the source")
        except re.error as e:
            errors.append(f"{sid}: bad regex {f['pattern']!r}: {e}")
    exp = it.get("expected") or {}
    for i in ids:
        if not is_hit(chunks[i], exp):
            errors.append(f"{sid}: expected location does not cover source chunk {i}")
    for lang in ("en", "fr"):
        q = it[f"question_{lang}"]
        cr = copy_ratio(q, source)
        if cr > spec["review"]["max_copy_ratio"]:
            warnings.append(
                f"{sid}: question_{lang} copy {cr} > {spec['review']['max_copy_ratio']}"
            )
        if YES_NO.match(q) or META.search(q):
            warnings.append(f"{sid}: question_{lang} is yes/no or mentions passage/article")
    if gt == " ".join(draft["ground_truth"].split()):
        warnings.append(f"{sid}: ground_truth unchanged from the draft - re-read it against SOURCE")
    return errors, warnings


def assign_languages(items: list[dict], chunks: dict, spec: dict) -> list[str]:
    """Fill question_language 'auto' to meet the FR/EN and cross-lingual targets (seeded)."""
    rv, rng = spec["review"], random.Random(spec["seed"])
    langs = {it["seed_id"]: {chunks[i]["language"] for i in it["source_chunk_ids"]} for it in items}
    work_langs: dict[str, set] = {}
    for c in chunks.values():
        work_langs.setdefault(c["work_id"], set()).add(c["language"])

    def is_cross(it, lang):
        return lang not in work_langs[it["expected"]["work_id"]]

    fixed = [it for it in items if it["question_language"] != "auto"]
    auto = sorted(
        (it for it in items if it["question_language"] == "auto"), key=lambda x: x["seed_id"]
    )
    rng.shuffle(auto)
    for it in fixed:
        it["_lang"] = it["question_language"]
    cross_need = rv["min_cross_lingual"] - sum(is_cross(it, it["_lang"]) for it in fixed)

    def src_lang(it):
        return sorted(langs[it["seed_id"]])[0]

    # 1) cross-lingual (monolingual works only), alternating directions FR->EN doc / EN->FR doc
    mono = [it for it in auto if len(work_langs[it["expected"]["work_id"]]) == 1]
    by_src = {lang: [it for it in mono if src_lang(it) == lang] for lang in ("en", "fr")}
    turn = "en"
    while cross_need > 0 and (by_src["en"] or by_src["fr"]):
        turn = turn if by_src[turn] else ("fr" if turn == "en" else "en")
        it = by_src[turn].pop(0)
        it["_lang"] = "fr" if turn == "en" else "en"
        cross_need -= 1
        turn = "fr" if turn == "en" else "en"
    # 2) everything else in its source language (AI Act: either language is same-language)
    for it in auto:
        if "_lang" not in it:
            it["_lang"] = src_lang(it) if len(work_langs[it["expected"]["work_id"]]) == 1 else "en"
    # 3) meet the FR target: flip bilingual-work items first (no extra cross-lingual), then mono
    target = rv["languages"]["fr"]
    rest = [it for it in auto if not is_cross(it, it["_lang"])]
    flips = sorted(rest, key=lambda it: len(work_langs[it["expected"]["work_id"]]) == 1)
    for it in flips:
        n_fr = sum(x["_lang"] == "fr" for x in items)
        if n_fr < target and it["_lang"] == "en":
            it["_lang"] = "fr"
        elif n_fr > target and it["_lang"] == "fr":
            it["_lang"] = "en"
    errors = []
    final = Counter(it["_lang"] for it in items)
    if dict(final) != {k: v for k, v in rv["languages"].items() if v}:
        errors.append(f"languages {dict(final)} != target {rv['languages']} (fix fixed choices)")
    n_cross = sum(is_cross(it, it["_lang"]) for it in items)
    if n_cross < rv["min_cross_lingual"]:
        errors.append(f"only {n_cross} cross-lingual < {rv['min_cross_lingual']}")
    return errors


def apply_sheet(cfg: dict, spec: dict) -> None:
    path = PROJECT_ROOT / spec["review"]["sheet"]
    try:
        items = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:  # e.g. an unclosed quote: show where, not a traceback
        raise SystemExit(f"YAML syntax error in the sheet (nothing checked):\n{e}") from None
    drafts = latest_drafts(spec)
    chunks = {c["chunk_id"]: c for c in load_chunks(cfg)}
    status = Counter(it["decision"] for it in items)
    errors, warnings = [], []
    if set(status) - {"todo", "accept", "drop"}:
        errors.append(f"unknown decision values: {set(status) - {'todo', 'accept', 'drop'}}")
    for it in items:
        if it["decision"] == "drop" and not it.get("drop_reason"):
            errors.append(f"{it['seed_id']}: drop needs a drop_reason")
    accepted = [it for it in items if it["decision"] == "accept"]
    for it in accepted:
        e, w = check_item(it, drafts[it["seed_id"]]["draft"], chunks, spec)
        errors += e
        warnings += w
    need = needed_per_work(spec)
    have = Counter(it["expected"]["work_id"] for it in accepted)
    seed_kind = {
        s["seed_id"]: s["kind"]
        for s in map(json.loads, open(PROJECT_ROOT / spec["sampling"]["output"], encoding="utf-8"))
    }
    for work, w in spec["works"].items():  # min_kind is for all answerable questions: warn only
        for kind, k in w.get("min_kind", {}).items():
            got = sum(
                1
                for it in accepted
                if it["expected"]["work_id"] == work and seed_kind[it["seed_id"]] == kind
            )
            if got < k:
                warnings.append(
                    f"{work}: {got} {kind} accepted < min_kind {k} (or cover it in 5.5)"
                )
    print(f"decisions: {dict(status)}")
    print(f"{'work':<17}{'accepted':>9}{'needed':>7}")
    for w, n in need.items():
        print(f"{w:<17}{have[w]:>9}{n:>7}{'' if have[w] == n else '  <-'}")
    complete = have == Counter({w: n for w, n in need.items() if n})
    if complete and not errors:
        errors += assign_languages(accepted, chunks, spec)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors or not complete:
        print("\nnot written: fix the errors / finish the decisions, then --apply again")
        raise SystemExit(1)
    rows = []
    for it in accepted:
        d = drafts[it["seed_id"]]
        lang = it["_lang"]
        draft = d["draft"]
        rows.append(
            {
                "seed_id": it["seed_id"],
                "type": "factual",
                "origin": "llm_7b",
                "reviewed": True,
                # 5.4-B: Claude rewrote; high-risk items verified line by line, rest spot-checked
                "verification": "line_by_line"
                if "HIGH-RISK verified line by line" in (it.get("notes") or "")
                else "spot_check",
                "edited": {
                    "question": " ".join(it[f"question_{lang}"].split())
                    != " ".join(draft[f"question_{lang}"].split()),
                    "ground_truth": " ".join(it["ground_truth"].split())
                    != " ".join(draft["ground_truth"].split()),
                    "key_facts": [f["fact"] for f in it["key_facts"]] != draft["key_facts"],
                },
                "question": " ".join(it[f"question_{lang}"].split()),
                "question_language": lang,
                "expected_behavior": "answer",
                "expected": it["expected"],
                "source_chunk_ids": it["source_chunk_ids"],
                "ground_truth": " ".join(it["ground_truth"].split()),
                "key_facts": it["key_facts"],
                "notes": it.get("notes"),
                "draft": {"model": d["model"], "prompt_version": d["prompt_version"]},
            }
        )
    out = PROJECT_ROOT / spec["review"]["accepted"]
    write_jsonl(rows, out)
    print(
        f"\n{len(rows)} accepted -> {spec['review']['accepted']}; "
        f"languages {dict(Counter(r['question_language'] for r in rows))}"
    )


def show(cfg: dict, chunk_id: str, n: int) -> None:
    chunks = load_chunks(cfg)
    pos = next((i for i, c in enumerate(chunks) if c["chunk_id"] == chunk_id), None)
    if pos is None:
        raise SystemExit(f"unknown chunk_id {chunk_id!r} (format: <doc_id>::structure::NNNN)")
    doc = chunks[pos]["doc_id"]
    for c in chunks[max(0, pos - n) : pos + n + 1]:
        if c["doc_id"] == doc:
            mark = ">>" if c["chunk_id"] == chunk_id else "  "
            print(f"{mark} {c['chunk_id']}  (article {c['article']}, {c['n_tokens']} tokens)")
            print(textwrap.indent(c["text"], "   ") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make-sheet", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", metavar="CHUNK_ID")
    ap.add_argument("-n", type=int, default=0, help="neighbours on each side for --show")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_spec(cfg)
    if args.make_sheet:
        make_sheet(cfg, spec, args.force)
    elif args.apply:
        apply_sheet(cfg, spec)
    elif args.show:
        show(cfg, args.show, args.n)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
