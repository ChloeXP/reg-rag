"""Step 6.9 - golden-v2 patches: applied as declared, and the audit cases as regression tests.

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
    ],
)
def test_transitional_ranges_no_longer_count_as_the_minimum(fact, text):
    assert not matches(pattern("G003", fact), text)


def test_a_patch_never_applies_to_a_changed_source():
    items = [{"src_id": "X1", "key_facts": [{"fact": "f", "pattern": "current"}], "expected": []}]
    patches = {"patches": [{"id": "T-1", "src_id": "X1", "fact": "f", "old": "stale", "new": "n"}]}
    assert apply_patches(items, patches) and items[0]["key_facts"][0]["pattern"] == "current"
