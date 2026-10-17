"""End-to-end tests for step 4: rag.ask on the real corpus with the baseline LLM (qwen2.5:3b).

Marked `ollama`; skipped when Ollama or Qdrant is not reachable (e.g. CI). ~45 s in total.
Known baseline failures are xfail on purpose (journal P39/P40): they document the limits of
the baseline and turn into XPASS when a later step fixes them.
"""

import re

import httpx
import pytest
import yaml

from rag.ask import ask
from rag.config import PROJECT_ROOT, load_config

CFG = load_config()
QUERIES = {
    q["id"]: q
    for q in yaml.safe_load((PROJECT_ROOT / "eval" / "smoke_queries.yaml").read_text("utf-8"))
}


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not _up(f"{CFG['llm']['base_url']}/api/tags"), reason="Ollama not running"),
    pytest.mark.skipif(not _up(f"{CFG['vector_store']['url']}/healthz"), reason="Qdrant not up"),
]


def run(qid: str) -> dict:
    return ask(QUERIES[qid]["query"], CFG)  # ask() never writes the run log itself


def test_q1_french_answer_cites_dora_article_19():
    rec = run("Q1")
    assert rec["language"] == "fr" and not rec["is_refusal"]
    assert rec["invalid"] == [] and not rec["uncited"]
    assert any(r["work_id"] == "dora" and r["article"] == "19" for r in rec["references"])
    assert rec["config"]["prompt_version"] == "v1" and rec["prompt_tokens"] > 1000


@pytest.mark.xfail(reason="P39: CRR Art. 395 para 1 (25 % limit) is not in the top-5 chunks")
def test_q4_answer_states_the_25_percent_limit():
    assert re.search(r"25\s*%", run("Q4")["answer"])


@pytest.mark.xfail(reason="P40: 3b answers '[1] defines high-risk AI systems' without the article")
def test_q5_answer_names_article_6():
    assert "Article 6" in run("Q5")["answer"]
