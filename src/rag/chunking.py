"""Chunking (steps 2.5-2.6): list[Block] -> list[Chunk].

2.5 fixed strategy: glue all content blocks of a document into one text, then cut it into
windows of `chunk_size` tokens with `chunk_overlap` tokens shared between neighbours.
Tokens are counted with the embedding model's own tokenizer (BAAI/bge-m3, decision 2.5-A).
It ignores structure on purpose: it is the baseline the structure strategy must beat.

Debug run: uv run python -m rag.chunking
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from rag.config import PROJECT_ROOT, load_config
from rag.parse_pdf import parse_pdf
from rag.parse_xhtml import parse_xhtml
from rag.schema import Block, Chunk, make_chunk_id

CONTENT_KINDS = {"title", "recital", "article", "annex", "body", "table"}
WORD_START = "\u2581"  # "▁": SentencePiece marks the first piece of every word with it
MAX_SNAP = 16  # how many tokens we may step back to avoid cutting a word in half


@lru_cache(maxsize=2)
def get_tokenizer(name: str) -> Tokenizer:
    """Download once from the Hugging Face Hub (then cached locally)."""
    return Tokenizer.from_pretrained(name)


def count_tokens(text: str, cfg: dict) -> int:
    tok = get_tokenizer(cfg["chunking"]["tokenizer"])
    return len(tok.encode(text, add_special_tokens=False).ids)


def load_manifest(cfg: dict) -> dict[str, dict]:
    """doc_id -> manifest row (work_id, language, doc_type, format...): the single source."""
    with open(PROJECT_ROOT / cfg["paths"]["manifest"], encoding="utf-8") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def load_blocks(path: Path, cfg: dict) -> list[Block]:
    """Dispatch on file format: one parser per format, same Block output (decision #22)."""
    if path.suffix == ".pdf":
        return parse_pdf(path, cfg)[0]
    if path.suffix == ".xhtml":
        return parse_xhtml(path)[0]
    raise ValueError(f"no parser for {path.name}")


def content_blocks(blocks: list[Block], cfg: dict) -> list[Block]:
    kinds = CONTENT_KINDS | ({"footnote"} if cfg["chunking"]["include_footnotes"] else set())
    return [b for b in blocks if b.kind in kinds]


def fixed_windows(blocks: list[Block], cfg: dict) -> list[tuple[str, list[Block], int]]:
    """Sliding token window over the whole document -> (text, blocks it covers, n_tokens)."""
    size, overlap = cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"]
    tok = get_tokenizer(cfg["chunking"]["tokenizer"])

    # 1) one text for the whole document, remembering where each block sits in it
    text, spans = "", []
    for b in blocks:
        spans.append((len(text), len(text) + len(b.text)))
        text += b.text + "\n\n"

    # 2) tokenize once; offsets map every token back to characters of `text`
    enc = tok.encode(text, add_special_tokens=False)
    tokens, offsets, n = enc.tokens, enc.offsets, len(enc.tokens)

    windows = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:  # step back to a word start, so we never cut "superv|ision"
            for k in range(end, max(start + size - MAX_SNAP, start + overlap + 1), -1):
                if tokens[k].startswith(WORD_START):
                    end = k
                    break
        c0, c1 = offsets[start][0], offsets[end - 1][1]
        covered = [b for b, (s, e) in zip(blocks, spans, strict=True) if s < c1 and e > c0]
        windows.append((text[c0:c1].strip(), covered, end - start))
        if end >= n:
            break
        start = end - overlap
        for k in range(start, min(start + MAX_SNAP, end)):  # also START on a word start
            if tokens[k].startswith(WORD_START):
                start = k
                break
    return windows


def fixed_chunks(blocks: list[Block], meta: dict, cfg: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, (text, covered, n_tokens) in enumerate(
        fixed_windows(content_blocks(blocks, cfg), cfg)
    ):
        pages = [b.page for b in covered if b.page is not None]
        first = covered[0]
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(meta["doc_id"], "fixed", index),
                doc_id=meta["doc_id"],
                work_id=meta["work_id"],
                language=meta["language"],
                doc_type=meta["doc_type"],
                format=meta["format"],
                strategy="fixed",
                index=index,
                text=text,
                n_tokens=n_tokens,
                kind=first.kind,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                article=first.article,  # where the chunk STARTS (it may span several)
                section=first.section,
            )
        )
    return chunks


def main() -> None:
    """Debug: per document, token and chunk stats + how often a fixed chunk mixes articles."""
    cfg = load_config()
    manifest = load_manifest(cfg)
    raw = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    totals: dict[str, int] = {}
    for doc_id, meta in manifest.items():
        blocks = content_blocks(load_blocks(raw / meta["filename"], cfg), cfg)
        windows = fixed_windows(blocks, cfg)
        totals[doc_id] = sum(count_tokens(b.text, cfg) for b in blocks)
        has_articles = any(b.article for b in blocks)
        mixed = sum(len({b.article for b in cov if b.article}) > 1 for _, cov, _ in windows)
        sizes = [n for _, _, n in windows]
        print(
            f"{doc_id:30} tokens={totals[doc_id]:>7} chunks={len(windows):>4} "
            f"min/max={min(sizes)}/{max(sizes)}"
            + (f"  chunks mixing 2+ articles={mixed}/{len(windows)}" if has_articles else "")
        )
    en, fr = totals.get("ai_act_en"), totals.get("ai_act_fr")
    if en and fr:
        print(f"\nAI Act tokens: EN={en}  FR={fr}  FR/EN={fr / en:.2f}")

    meta = manifest["ai_act_en"]
    sample = fixed_chunks(load_blocks(raw / meta["filename"], cfg), meta, cfg)[120]
    print(
        f"\nsample {sample.chunk_id}: {sample.n_tokens} tokens, pages "
        f"{sample.page_start}-{sample.page_end}, starts in article {sample.article}"
    )
    print(f"  starts: {sample.text[:110]!r}")
    print(f"  ends:   {sample.text[-110:]!r}")


if __name__ == "__main__":
    main()
