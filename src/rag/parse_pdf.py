"""PDF parser (step 2.2): raw PDF -> list[Block].

2.2a: read text line by line with positions, and detect boilerplate (headers/footers)
      by repetition across pages instead of a fixed % margin (fails on landscape pages).
2.2b: pull footnotes out of the body text so they no longer cut sentences across pages.

Debug run: uv run python -m rag.parse_pdf
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from rag.config import PROJECT_ROOT, load_config
from rag.schema import Block

# A footnote starts with "(12)" (EU Official Journal) or "12 " / "12" (EBA, ACPR) or "*".
FOOTNOTE_MARK = re.compile(r"^\s*(\(\d+\)|\d+\s|\d+$|\*)")


@dataclass(frozen=True)
class Line:
    page: int  # 1-based
    y0: float  # top of the line, in points from the top of the page
    y1: float  # bottom of the line
    page_height: float
    text: str
    size: float  # largest font size on the line (useful for headings, footnotes)


@dataclass(frozen=True)
class Footnote:
    page: int
    text: str


# ---------- 2.2a: lines + boilerplate ----------


def iter_lines(doc: pymupdf.Document) -> Iterator[Line]:
    """Yield every non-empty text line of the document, page by page, top to bottom."""
    for page in doc:
        height = page.rect.height
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:  # 0 = text, 1 = image
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                if not text.strip():
                    continue
                _x0, y0, _x1, y1 = line["bbox"]
                size = max(span["size"] for span in line["spans"])
                yield Line(page.number + 1, y0, y1, height, text, size)


def pattern(text: str) -> str:
    """Mask digits and collapse spaces: 'Page 3' and 'Page 4' -> same pattern 'Page #'."""
    return re.sub(r"\d+", "#", " ".join(text.split()))


def edge_zone(line: Line, band_pt: float) -> str | None:
    """'top' / 'bottom' if the line sits within band_pt of the page edge, else None."""
    if line.y1 <= band_pt:
        return "top"
    if line.y0 >= line.page_height - band_pt:
        return "bottom"
    return None


def find_boilerplate(lines: list[Line], n_pages: int, cfg: dict) -> set[tuple[str, str]]:
    """Return (zone, pattern) pairs that repeat in the edge zone on enough pages."""
    band = cfg["parsing"]["edge_band_pt"]
    min_ratio = cfg["parsing"]["boilerplate_min_ratio"]
    pages_seen: dict[tuple[str, str], set[int]] = {}
    for line in lines:
        zone = edge_zone(line, band)
        if zone:
            pages_seen.setdefault((zone, pattern(line.text)), set()).add(line.page)
    return {key for key, pages in pages_seen.items() if len(pages) / n_pages >= min_ratio}


def is_boilerplate(line: Line, boilerplate: set[tuple[str, str]], band_pt: float) -> bool:
    zone = edge_zone(line, band_pt)
    return zone is not None and (zone, pattern(line.text)) in boilerplate


# ---------- 2.2b: footnotes ----------


def body_font_size(page_lines: list[Line]) -> float:
    """The font size carrying the most characters on this page = body text size."""
    chars = Counter()
    for line in page_lines:
        chars[round(line.size, 1)] += len(line.text)
    return chars.most_common(1)[0][0]


def split_footnotes(page_lines: list[Line], gap: float) -> tuple[list[Line], list[Line]]:
    """Split one page into (body lines, footnote lines).

    Footnote zone = the run of lines at the BOTTOM of the page whose font is smaller than
    the page's body font, and whose first line starts with a footnote mark. Looking only
    at the bottom run protects small-font text in the middle of the page (e.g. recital
    numbers "(15)" in the EU Official Journal use the same size as footnote marks).
    """
    if not page_lines:
        return [], []
    body_size = body_font_size(page_lines)
    start = len(page_lines)
    while start > 0 and page_lines[start - 1].size < body_size - gap:
        start -= 1
    # The zone must begin with a footnote mark; drop leading lines until it does.
    while start < len(page_lines) and not FOOTNOTE_MARK.match(page_lines[start].text):
        start += 1
    return page_lines[:start], page_lines[start:]


PAREN_MARK = re.compile(r"^\s*\(\d+\)")  # "(12) ..." EU Official Journal style
BARE_MARK = re.compile(r"^\s*\d+(\s|$)")  # "12 ..." EBA / ACPR style


def group_footnotes(lines: list[Line]) -> list[Footnote]:
    """Merge footnote lines into one Footnote per mark: '(15)' + its continuation lines.

    Each document uses ONE numbering style. We detect it by majority vote, then only that
    style may open a new footnote. Otherwise a continuation line such as "21 May 2024."
    inside an "(N)"-style footnote would wrongly be read as footnote 21.
    """
    n_paren = sum(bool(PAREN_MARK.match(ln.text)) for ln in lines)
    n_bare = sum(bool(BARE_MARK.match(ln.text)) for ln in lines)
    mark = PAREN_MARK if n_paren >= n_bare else BARE_MARK

    notes: list[Footnote] = []
    for line in lines:
        text = line.text.strip()
        if mark.match(line.text) or not notes or notes[-1].page != line.page:
            notes.append(Footnote(line.page, text))
        else:
            notes[-1] = Footnote(line.page, f"{notes[-1].text} {text}")
    return notes


# ---------- cleaning pipeline for one document ----------


def clean_document(path: Path, cfg: dict) -> tuple[list[Line], list[Footnote], dict]:
    """Return (body lines, footnotes, stats) for one PDF."""
    band = cfg["parsing"]["edge_band_pt"]
    gap = cfg["parsing"]["footnote_size_gap"]
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        lines = list(iter_lines(doc))

    boilerplate = find_boilerplate(lines, n_pages, cfg)
    kept = [ln for ln in lines if not is_boilerplate(ln, boilerplate, band)]

    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in kept:
        by_page[line.page].append(line)

    body: list[Line] = []
    footnote_lines: list[Line] = []
    for page in sorted(by_page):
        page_body, page_notes = split_footnotes(by_page[page], gap)
        body.extend(page_body)
        footnote_lines.extend(page_notes)

    footnotes = group_footnotes(footnote_lines)
    stats = {
        "pages": n_pages,
        "lines": len(lines),
        "boilerplate": len(lines) - len(kept),
        "footnote_lines": len(footnote_lines),
        "footnotes": len(footnotes),
        "body_lines": len(body),
    }
    return body, footnotes, stats


def footnote_numbering_gaps(footnotes: list[Footnote]) -> tuple[int, int, list[int]]:
    """Footnotes are numbered 1..N: gaps in the numbers we found = footnotes we missed.

    Returns (unique numbers found, highest number, missing numbers) -> a free recall check.
    """
    found = set()
    for note in footnotes:
        m = re.match(r"\s*\(?(\d+)", note.text)
        if m:
            found.add(int(m.group(1)))
    highest = max(found, default=0)
    return len(found), highest, [i for i in range(1, highest + 1) if i not in found]


# ---------- 2.2c: lines -> paragraphs -> Blocks ----------

# Text that ends a sentence (or a list item) - used to decide cross-page joins.
SENTENCE_END = re.compile(r"[.;:!?\u00bb\u201d)]\s*$")


def group_paragraphs(body: list[Line], gap_ratio: float) -> list[list[Line]]:
    """Group body lines into paragraphs.

    Same page: start a new paragraph when the vertical gap to the previous line is larger
    than gap_ratio x the previous line's height (normal line spacing is ~0.1-0.3).
    New page: continue the previous paragraph if it did not end a sentence
    (a sentence cut by a page break), otherwise start a new one.
    """
    paragraphs: list[list[Line]] = []
    for line in body:
        if not paragraphs:
            paragraphs.append([line])
            continue
        prev = paragraphs[-1][-1]
        if line.page == prev.page:
            is_new = (line.y0 - prev.y1) > gap_ratio * (prev.y1 - prev.y0)
        else:
            is_new = bool(SENTENCE_END.search(prev.text))
        if is_new:
            paragraphs.append([line])
        else:
            paragraphs[-1].append(line)
    return paragraphs


def join_lines(lines: list[Line]) -> str:
    """Join the lines of one paragraph into one string.

    A line ending with "-" is glued to the next line WITHOUT a space but KEEPING the hyphen:
    in these PDFs every line-final hyphen is a real compound ("risk-|based",
    "non-|executive", "sous-|traitance"); there is no typographic word splitting, so the
    usual "remove the hyphen" rule would corrupt words ("riskbased").
    """
    text = ""
    for line in lines:
        part = line.text.strip()
        if not text:
            text = part
        elif text.endswith("-"):
            text += part
        else:
            text += " " + part
    return " ".join(text.split())  # collapse repeated spaces


def parse_pdf(path: Path, cfg: dict) -> tuple[list[Block], dict]:
    """Raw PDF -> Blocks (body paragraphs, then footnotes) + cleaning stats."""
    doc_id = path.stem
    body, footnotes, stats = clean_document(path, cfg)
    paragraphs = group_paragraphs(body, cfg["parsing"]["paragraph_gap_ratio"])

    blocks: list[Block] = []
    for para in paragraphs:
        # page = page where the paragraph starts (a paragraph may continue on the next page)
        blocks.append(Block(doc_id, len(blocks), join_lines(para), "body", page=para[0].page))
    for note in footnotes:
        blocks.append(Block(doc_id, len(blocks), note.text, "footnote", page=note.page))

    stats["paragraphs"] = len(paragraphs)
    stats["cross_page_paragraphs"] = sum(p[0].page != p[-1].page for p in paragraphs)
    return blocks, stats


def main() -> None:
    """Debug: per PDF, show cleaning stats, footnote numbering check and sample paragraphs."""
    cfg = load_config()
    for path in sorted((PROJECT_ROOT / cfg["paths"]["raw_dir"]).glob("*.pdf")):
        blocks, s = parse_pdf(path, cfg)
        body = [b for b in blocks if b.kind == "body"]
        notes = [Footnote(b.page, b.text) for b in blocks if b.kind == "footnote"]
        n_found, highest, missing = footnote_numbering_gaps(notes)
        lengths = sorted(len(b.text) for b in body)
        print(
            f"\n=== {path.stem} === pages={s['pages']} boilerplate=-{s['boilerplate']} "
            f"paragraphs={s['paragraphs']} (cross-page={s['cross_page_paragraphs']}) "
            f"footnotes={len(notes)}"
        )
        print(
            f"  paragraph chars: median={lengths[len(lengths) // 2]} max={lengths[-1]}  "
            f"footnote numbering: {n_found}/{highest} missing={missing}"
        )
        for b in body[len(body) // 3 :: max(1, len(body) // 3)][:2]:
            print(f"  p.{b.page:<3} {b.text[:100]!r}")


if __name__ == "__main__":
    main()
