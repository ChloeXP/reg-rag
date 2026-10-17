"""Chunking (steps 2.5-2.6): list[Block] -> list[Chunk].

2.5 fixed strategy: glue all content blocks of a document into one text, then cut it into
windows of `chunk_size` tokens with `chunk_overlap` tokens shared between neighbours.
Tokens are counted with the embedding model's own tokenizer (BAAI/bge-m3, decision 2.5-A).
It ignores structure on purpose: it is the baseline the structure strategy must beat.

2.6 structure strategy (decision 2.6-B): one legal unit (article, recital, annex part,
guideline section) never shares a chunk with another unit. A unit longer than
`chunk_size` is split at paragraph boundaries; every chunk starts with a context header
"[CRR · Article 92 Own funds requirements]" so a split-off part still says what it is.

Debug run: uv run python -m rag.chunking
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from rag.config import PROJECT_ROOT, load_config
from rag.parse_pdf import load_sources, parse_pdf
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
        skip = load_sources(cfg).get(path.stem, {}).get("skip_sections") or []
        return parse_xhtml(path, set(skip))[0]
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


# ---------- 2.6: structure strategy ----------


def unit_key(b: Block) -> tuple:
    """Which legal unit a block belongs to. Consecutive blocks with the same key = one unit."""
    if b.article:
        return ("article", b.article)
    if b.kind == "recital":
        return ("recital", b.order)  # every recital is its own unit
    if b.kind == "annex":
        return ("annex", b.section)
    return ("section", b.section)  # guidelines/reports: the heading path; None = preamble


def units_of(blocks: list[Block]) -> list[list[Block]]:
    """Group content blocks into units, in reading order. Structural headings (chapter,
    section, EBA titles) are not content: they live in the section path / header."""
    units: list[list[Block]] = []
    for b in blocks:
        if b.kind == "title" and not b.article:
            continue
        if units and unit_key(units[-1][-1]) == unit_key(b):
            units[-1].append(b)
        else:
            units.append([b])
    return units


def unit_header(unit: list[Block], meta: dict) -> str:
    """Context header written at the top of every chunk of the unit."""
    doc = meta["work_id"].replace("_", " ").upper()
    first = unit[0]
    if first.article:
        title = next((b.text for b in unit if b.kind == "title"), f"Article {first.article}")
        return f"[{doc} · {title}]"
    if first.kind == "recital":
        num = first.text.split(")")[0].lstrip("(")
        word = "Considérant" if meta["language"] == "fr" else "Recital"
        return f"[{doc} · {word} {num}]"
    path = (first.section or "").split(" > ")[-2:]  # last two levels are enough
    return f"[{doc} · {' > '.join(path)}]" if path != [""] else f"[{doc}]"


def pack_unit(unit: list[Block], header: str, cfg: dict) -> list[tuple[str, list[Block]]]:
    """Greedy packing of a unit's paragraphs into chunks of at most chunk_size tokens,
    cutting only between paragraphs. A single paragraph that is too long is cut with the
    fixed token window (the only case where a cut can fall inside a paragraph)."""
    limit = cfg["chunking"]["chunk_size"]
    body = [b for b in unit if not (b.kind == "title" and b.article)]  # title is in header
    if not body:
        body = unit
    pieces: list[tuple[str, list[Block]]] = []
    current: list[Block] = []

    def text_of(blocks: list[Block]) -> str:
        return header + "\n" + "\n".join(b.text for b in blocks)

    for b in body:
        if count_tokens(text_of([*current, b]), cfg) <= limit:
            current.append(b)
            continue
        if current:
            pieces.append((text_of(current), current))
            current = []
        if count_tokens(text_of([b]), cfg) <= limit:
            current = [b]
            continue
        # one paragraph alone is too long: token windows inside it, header on each part.
        # Re-tokenizing "header + part" can add a token or two at the join (or around table
        # line breaks), so shrink the window until every part fits.
        room = limit - count_tokens(header + "\n", cfg)
        while True:
            sub_cfg = {**cfg, "chunking": {**cfg["chunking"], "chunk_size": room}}
            parts = [header + "\n" + t for t, _, _ in fixed_windows([b], sub_cfg)]
            if all(count_tokens(p, cfg) <= limit for p in parts):
                break
            room -= 4
        pieces.extend((p, [b]) for p in parts)
    if current:
        pieces.append((text_of(current), current))
    return pieces


def structure_chunks(blocks: list[Block], meta: dict, cfg: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units_of(content_blocks(blocks, cfg)):
        header = unit_header(unit, meta)
        for text, covered in pack_unit(unit, header, cfg):
            assert len({unit_key(b) for b in covered}) == 1, "a chunk must hold ONE unit"
            pages = [b.page for b in covered if b.page is not None]
            first = covered[0]
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(meta["doc_id"], "structure", index),
                    doc_id=meta["doc_id"],
                    work_id=meta["work_id"],
                    language=meta["language"],
                    doc_type=meta["doc_type"],
                    format=meta["format"],
                    strategy="structure",
                    index=index,
                    text=text,
                    n_tokens=count_tokens(text, cfg),
                    kind=first.kind,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    article=first.article,  # exact: a structure chunk never spans 2 articles
                    section=first.section,
                )
            )
    return chunks


def main() -> None:
    """Debug: fixed vs structure, per document; then a look at CRR Article 92."""
    cfg = load_config()
    manifest = load_manifest(cfg)
    raw = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    limit = cfg["chunking"]["chunk_size"]
    print(f"{'doc_id':30} {'fixed':>6} {'struct':>6} {'median tok':>10} {'>limit':>6}")
    for doc_id, meta in manifest.items():
        blocks = load_blocks(raw / meta["filename"], cfg)
        n_fixed = len(fixed_windows(content_blocks(blocks, cfg), cfg))
        chunks = structure_chunks(blocks, meta, cfg)
        sizes = sorted(c.n_tokens for c in chunks)
        over = sum(n > limit for n in sizes)
        print(f"{doc_id:30} {n_fixed:>6} {len(chunks):>6} {sizes[len(sizes) // 2]:>10} {over:>6}")

    meta = manifest["crr_en"]
    chunks = structure_chunks(load_blocks(raw / meta["filename"], cfg), meta, cfg)
    art92 = [c for c in chunks if c.article == "92"]
    art4 = [c for c in chunks if c.article == "4"]
    print(f"\nCRR Article 4 (definitions) -> {len(art4)} chunks; Article 92 -> {len(art92)}")
    print(f"--- {art92[0].chunk_id} ({art92[0].n_tokens} tokens) ---")
    print(art92[0].text[:700])


if __name__ == "__main__":
    main()
