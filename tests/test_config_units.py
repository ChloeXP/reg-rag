"""Step 6.0 - config layout: build-time chunking params vs query-time retrieval params."""

import copy

import pytest

from rag.config import apply_overrides, load_config

CFG = load_config()


def test_strategy_lives_under_retrieval():
    # 4.1-C: the strategy is chosen at query time; chunking.* only describes how chunks are built
    assert CFG["retrieval"]["strategy"] in {"fixed", "structure"}
    assert "strategy" not in CFG["chunking"]


def test_overrides_parse_yaml_values_on_a_copy():
    cfg = apply_overrides(copy.deepcopy(CFG), ["retrieval.mode=hybrid", "retrieval.top_k=10"])
    assert cfg["retrieval"]["mode"] == "hybrid" and cfg["retrieval"]["top_k"] == 10
    assert CFG["retrieval"]["top_k"] == 5  # the loaded config is untouched


@pytest.mark.parametrize("bad", ["retrieval.mod=hybrid", "retrival.mode=dense", "no_equals_sign"])
def test_override_typos_fail_loudly(bad):
    with pytest.raises((KeyError, ValueError)):
        apply_overrides(copy.deepcopy(CFG), [bad])


def test_eval_depth_covers_ks_and_hybrid_candidates():
    e = CFG["eval"]
    assert max(e["ks"]) <= e["retrieval_depth"] <= CFG["retrieval"]["hybrid"]["candidates"]
    assert CFG["retrieval"]["top_k"] in e["ks"]
