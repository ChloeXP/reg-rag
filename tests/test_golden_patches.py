"""Steps 6.9 / 7.9 - golden patches: applied as declared, and the audit cases as regression tests.

Each case below is an answer fragment seen in the 6.9 audit (5 e2e runs, 155 answers): a correct
paraphrase that v1 missed (must now match) or a wrong context that v1 accepted (must not).
"""

import json

import pytest
import yaml

from rag.answer_metrics import matches
from rag.build_golden import apply_patches
from rag.config import PROJECT_ROOT, load_config

CFG = load_config()
ROWS = {g["id"]: g for g in map(json.loads, (PROJECT_ROOT / CFG["eval"]["golden_set"]).open())}
PATCHES = yaml.safe_load((PROJECT_ROOT / CFG["eval"]["golden_patches"]).read_text("utf-8"))


def pattern(gid: str, fact: str) -> str:
    return next(k["pattern"] for k in ROWS[gid]["key_facts"] if k["fact"] == fact)


def test_every_row_carries_the_version_and_patched_rows_list_their_patches():
    assert {r["golden_version"] for r in ROWS.values()} == {PATCHES["version"]}
    applied = sorted(p for r in ROWS.values() for p in r.get("patches", []))
    assert applied == sorted(p["id"] for p in PATCHES["patches"])


@pytest.mark.parametrize(
    "gid, fact, text",
    [
        ("G007", "format lisible par machine", "marquées de manière machine-lisible"),
        ("G012", "méthodes explicatives", "the integration of explainable methods"),
        ("G015", "materially impair", "affecterait significativement"),
        ("G015", "materially impair", "compromettrait de manière matérielle"),
        ("G016", "interdependencies", "dependencies on other internal and external systems"),
        ("G029", "contrôle humain", "efficacement supervisés par des personnes naturelles"),
        ("G029", "contrôle humain", "une surveillance efficace par des personnes naturelles"),
        ("G030", "written contract", "clearly defined and documented in writing"),
        ("G028", "remains fully responsible", "l'institution reste responsable des activités"),
        ("G041", "regardless of the type of entity", "whatever the type of entity carrying out"),
        ("G003", "CET1 4,5 %", "un ratio Common Equity Tier 1 de 4,5 %"),
        ("G003", "Tier 1 6 %", "un ratio Tier 1 de 6 %"),
        # golden-v3 (7.9 audit of v1 / v2 / v3 answers)
        ("G013", "envisagé au sein de l'ACPR", "soit chez le superviseur (ACPR), soit dans des"),
        ("G013", "envisagé au sein de l'ACPR", "formés soit chez le superviseur lui-même"),
        (
            "G011",
            "approuvé par l'autorité compétente",
            "obtenir l'accord des autorités de contrôle",
        ),
        ("G012", "méthodes explicatives", "empirical methods that use explainability techniques"),
        ("G024", "not directly to IRB models", "cela ne s'applique pas directement aux modèles NI"),
        ("G024", "not directly to IRB models", "the AI Act does not apply directly to IRB models"),
    ],
)
def test_audited_paraphrases_now_match(gid, fact, text):
    assert matches(pattern(gid, fact), text)


@pytest.mark.parametrize(
    "fact, text",
    [
        ("CET1 4,5 %", "peuvent être compris entre 4% et 4,5% (Article 465)"),
        ("CET1 4,5 %", "un ratio de 4–4,5 % pendant la période transitoire"),
        ("Tier 1 6 %", "les ratios Tier 1 entre 5,5% et 6%"),
        ("CET1 4,5 %", "un ratio de fonds propres de 4 % à 4,5 % pour la Tier 1"),  # P80 (v2, v3)
        ("Tier 1 6 %", "et de 5,5 % à 6 % pour la Common Equity Tier 1"),  # P80
        ("CET1 4,5 %", "a range of 4 % to 4,5 %"),
    ],
)
def test_transitional_ranges_no_longer_count_as_the_minimum(fact, text):
    assert not matches(pattern("G003", fact), text)


def test_no_key_fact_is_satisfied_by_echoing_the_question_except_documented_ones():
    """7.9: a pattern that matches its own question can be scored by repeating the question."""
    allowed = {
        ("G008", "residential mortgages"),  # dev: answers state it as the asset type (audited)
        ("G028", "remains fully responsible"),  # dev: V2-07 BORDERLINE, "reste responsable"
        # test split: found by this check (question text only, no model answer was read);
        # to be reviewed before the single final test run
        ("G044", "10 days"),
        ("G047", "not become empty shells or letter-box entities"),
        ("G048", "recovery time objective"),
        ("G048", "recovery point objective"),
    }
    echo = {(g["id"], k["fact"]) for g in ROWS.values() for k in g.get("key_facts", [])
            if matches(k["pattern"], g["question"])}  # fmt: skip
    assert echo <= allowed, echo - allowed


def test_a_patch_never_applies_to_a_changed_source():
    items = [{"src_id": "X1", "key_facts": [{"fact": "f", "pattern": "current"}], "expected": []}]
    patches = {"patches": [{"id": "T-1", "src_id": "X1", "fact": "f", "old": "stale", "new": "n"}]}
    assert apply_patches(items, patches) and items[0]["key_facts"][0]["pattern"] == "current"
