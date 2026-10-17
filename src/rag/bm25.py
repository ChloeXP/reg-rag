"""BM25 keyword retrieval as Qdrant sparse vectors (step 3.5, decision 3.5-B).

BM25(q, d) = sum_t IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
            with tf = count of t in d, |d| = tokens in d, avgdl = mean |d| in the collection

Split of the work:
  - document side (here): the tf part with length normalisation -> sparse vector values
  - query side (here):    every query term gets weight 1.0
  - IDF (Qdrant):         collection created with Modifier.IDF, computed from the stored points
So the dot product sparse(q) . sparse(d) computed by Qdrant is exactly the BM25 score.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

TOKEN = re.compile(r"[a-z0-9]+")

# Small FR+EN function-word list (accents already stripped). Legal terms are never removed;
# frequent ones ("article", "institutions") are down-weighted by IDF instead.
STOPWORDS = frozenset(
    """
    the of and to in an for on by with as at or be is are was were been shall which that this
    these those from its it their them they not any such where when other may under into than
    what does do did say says who how whether there has have had all also each
    le la les de des du un une et en au aux par pour sur dans que qui est sont ou ne pas se ce
    cette ces son sa ses leur leurs il ils elle elles lui dont quel quelle quels quelles
    doivent doit ont avec sans entre comme plus lors selon etre
    """.split()
)


def normalize(text: str) -> str:
    """Lowercase + strip accents: 'Déclaration' -> 'declaration' (FR queries without accents)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Words and numbers; numbers are kept on purpose ('395' is the whole point of BM25 here)."""
    return [
        t
        for t in TOKEN.findall(normalize(text))
        if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
    ]


def token_id(token: str) -> int:
    """Stable 32-bit id for a token (sparse vector index). No vocabulary file to keep in sync."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")


def _sparse(counts: Counter) -> tuple[list[int], list[float]]:
    ids: Counter = Counter()
    for tok, value in counts.items():  # merges the (very rare) hash collisions
        ids[token_id(tok)] += value
    items = sorted(ids.items())
    return [i for i, _ in items], [float(v) for _, v in items]


def doc_vectors(
    texts: Sequence[str], k1: float, b: float
) -> tuple[list[tuple[list[int], list[float]]], float]:
    """Sparse BM25 document vectors (tf part only) + avgdl of this collection."""
    tokenized = [tokenize(t) for t in texts]
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    vectors = []
    for toks in tokenized:
        norm = k1 * (1 - b + b * len(toks) / avgdl)
        tf = Counter(toks)
        vectors.append(_sparse(Counter({t: n * (k1 + 1) / (n + norm) for t, n in tf.items()})))
    return vectors, avgdl


def query_vector(text: str) -> tuple[list[int], list[float]]:
    """Each distinct query term weighs 1.0 (Qdrant multiplies by IDF)."""
    return _sparse(Counter(dict.fromkeys(tokenize(text), 1.0)))
