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
import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.schema import Block
from rag.structure import Para, assign_structure

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
    bold: bool = False  # every span on the line is bold
    italic: bool = False  # every span on the line is italic


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
                spans = [s for s in line["spans"] if s["text"].strip()]
                # flag bit 16 = bold; some PDFs only say it in the font name
                bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans)
                italic = all(s["flags"] & 2 for s in spans)  # flag bit 2 = italic
                yield Line(page.number + 1, y0, y1, height, text, size, bold, italic)


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


# ---------- 2.4a: declared page exclusions (sources.yaml) ----------


def parse_page_ranges(spec: str | None) -> set[int]:
    """'2, 30-129' -> {2, 30, 31, ..., 129}. 1-based, inclusive. None/'' -> empty set."""
    pages: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        start, _, end = part.partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return pages


def load_sources(cfg: dict) -> dict[str, dict]:
    """doc_id -> its entry in data/sources.yaml (the hand-curated registry)."""
    path = PROJECT_ROOT / cfg["paths"]["sources"]
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]
    return {e["doc_id"]: e for e in entries}


# ---------- cleaning pipeline for one document ----------


def clean_document(
    path: Path, cfg: dict, skip: set[int] | None = None
) -> tuple[list[Line], list[Footnote], dict]:
    """Return (body lines, footnotes, stats) for one PDF, ignoring the `skip` pages."""
    band = cfg["parsing"]["edge_band_pt"]
    gap = cfg["parsing"]["footnote_size_gap"]
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        lines = list(iter_lines(doc))

    skip = skip or set()
    if skip and max(skip) > n_pages:
        raise ValueError(f"{path.name}: skip_pages goes to {max(skip)} but the PDF has {n_pages}")

    # Headers/footers are detected on ALL pages (they repeat on skipped pages too)
    boilerplate = find_boilerplate(lines, n_pages, cfg)
    kept = [ln for ln in lines if not is_boilerplate(ln, boilerplate, band)]

    # First real line of each skipped range, so a human can check the cut is in the right place
    starts = sorted(p for p in skip if p - 1 not in skip)
    first_line = {p: next((ln.text.strip() for ln in kept if ln.page == p), "") for p in starts}
    kept = [ln for ln in kept if ln.page not in skip]
    n_boilerplate = sum(1 for ln in lines if ln.page not in skip) - len(kept)
    lines = [ln for ln in lines if ln.page not in skip]
    n_pages -= len(skip)

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
        "pages": n_pages,  # pages kept
        "skipped_pages": len(skip),
        "skip_starts": first_line,
        "lines": len(lines),
        "boilerplate": n_boilerplate,
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


# ---------- 2.4b: headings ----------

# A full line that is a structural label in EU legal texts (EN + FR). Whole line only (P11).
LABEL = re.compile(
    r"(Article\s+(\d+|premier)|(CHAPTER|CHAPITRE|TITLE|TITRE|SECTION|Section)\s+[IVXLC\d]+"
    r"|ANNEXE?(\s+[IVXLC]+)?)"
)
# A line that OPENS a new heading (vs. the 2nd line of a multi-line heading)
HEADING_START = re.compile(r"\d+(\.\d+)*\.?\s*[A-ZÀ-Ý]|\d+(\.\d+)*\.?$|(Title|Annex)\s+\w+")


def body_size_of(lines: list[Line]) -> float:
    """Font size carrying the most characters in the whole document = body text size."""
    chars: Counter[float] = Counter()
    for line in lines:
        chars[round(line.size, 1)] += len(line.text)
    return chars.most_common(1)[0][0]


def is_label(line: Line) -> bool:
    """EU Official Journal structural label = the WHOLE line ("Article 6", "CHAPTER III").

    "Article N" must also be in italics: it is the one label that constantly appears as a
    cross-reference, and can land alone on a line when text wraps (AI Act Annex VIII) or
    when justified EBA text is split ("Article 131"). Official article labels are italic.
    """
    text = line.text.strip()
    if not LABEL.fullmatch(text):
        return False
    return line.italic or not text.startswith("Article")


def is_heading(line: Line, body_size: float, size_gap: float) -> bool:
    """Heading = an EU structural label, OR clearly bigger than body text, OR a short bold line.

    Three signals because publishers differ: EU Official Journal uses italic labels
    ("Article 6") + bold titles; EBA uses bigger blue fonts; ACPR uses bold Arial.
    """
    if len(line.text.strip()) > 150:
        return False
    return is_label(line) or line.size >= body_size + size_gap or line.bold


def opens_heading(line: Line) -> bool:
    return is_label(line) or bool(HEADING_START.match(line.text.strip()))


# ---------- 2.2c: lines -> paragraphs -> Blocks ----------

# Text that ends a sentence (or a list item) - used to decide cross-page joins.
SENTENCE_END = re.compile(r"[.;:!?\u00bb\u201d)]\s*$")


def group_paragraphs(
    body: list[Line], gap_ratio: float, heads: list[bool]
) -> list[tuple[list[Line], bool]]:
    """Group body lines into (paragraph lines, is_heading) pairs.

    Same page: start a new paragraph when the vertical gap to the previous line is larger
    than gap_ratio x the previous line's height (normal line spacing is ~0.1-0.3).
    New page: continue the previous paragraph if it did not end a sentence.
    Headings (2.4b) never merge with body text; consecutive heading lines merge
    ("Article 1" + "Subject matter") unless the new line opens a new heading.
    """
    paragraphs: list[tuple[list[Line], bool]] = []
    for line, head in zip(body, heads, strict=True):
        if not paragraphs:
            paragraphs.append(([line], head))
            continue
        prev_lines, prev_head = paragraphs[-1]
        prev = prev_lines[-1]
        if head or prev_head:
            is_new = not (head and prev_head and not opens_heading(line))
        elif line.page == prev.page:
            is_new = (line.y0 - prev.y1) > gap_ratio * (prev.y1 - prev.y0)
        else:
            is_new = bool(SENTENCE_END.search(prev.text))
        if is_new:
            paragraphs.append(([line], head))
        else:
            prev_lines.append(line)
    return paragraphs


NUMBER_GLUED = re.compile(r"^(\d+(?:\.\d+)*\.?)(?=[A-Za-zÀ-ÿ])")


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
    text = " ".join(text.split())  # collapse repeated spaces
    # EBA PDFs lose the space after numbers: '63.Institutions', '12.2Risk' -> add it back
    return NUMBER_GLUED.sub(r"\1 ", text)


def parse_pdf(path: Path, cfg: dict) -> tuple[list[Block], dict]:
    """Raw PDF -> Blocks (body paragraphs, then footnotes) + cleaning stats."""
    doc_id = path.stem
    skip = parse_page_ranges(load_sources(cfg).get(doc_id, {}).get("skip_pages"))
    body, footnotes, stats = clean_document(path, cfg, skip)
    body_size = body_size_of(body)
    gap = cfg["parsing"]["heading_size_gap"]
    heads = [is_heading(line, body_size, gap) for line in body]
    paragraphs = group_paragraphs(body, cfg["parsing"]["paragraph_gap_ratio"], heads)

    # 2.4c: place every paragraph in the document structure (kind, article, section)
    paras = [Para(join_lines(p), head, max(ln.size for ln in p)) for p, head in paragraphs]
    doc_type = load_sources(cfg).get(doc_id, {}).get("doc_type", "")
    places = assign_structure(paras, doc_type)

    blocks: list[Block] = []
    for (lines, _), para, (kind, article, section) in zip(paragraphs, paras, places, strict=True):
        # page = page where the paragraph starts (a paragraph may continue on the next page)
        blocks.append(Block(doc_id, len(blocks), para.text, kind, lines[0].page, article, section))
    for note in footnotes:
        blocks.append(Block(doc_id, len(blocks), note.text, "footnote", page=note.page))

    stats["paragraphs"] = len(paragraphs)
    stats["cross_page_paragraphs"] = sum(p[0].page != p[-1].page for p, _ in paragraphs)
    stats["body_size"] = body_size
    return blocks, stats


def main() -> None:
    """Debug: per PDF, structure check (2.4c) - kinds, recitals, articles, sample sections."""
    cfg = load_config()
    for path in sorted((PROJECT_ROOT / cfg["paths"]["raw_dir"]).glob("*.pdf")):
        blocks, s = parse_pdf(path, cfg)
        kinds = Counter(b.kind for b in blocks)
        print(f"\n=== {path.stem} === pages={s['pages']} blocks={len(blocks)} {dict(kinds)}")
        recitals = [
            int(m.group(1))
            for b in blocks
            if (m := re.match(r"\((\d+)\)", b.text)) and b.kind == "recital"
        ]
        if recitals:
            missing = [i for i in range(1, max(recitals) + 1) if i not in recitals]
            print(f"  recitals: {len(recitals)} numbered 1..{max(recitals)}, missing={missing}")
        labels = {b.article for b in blocks if b.kind == "title" and b.article}
        with_body = {b.article for b in blocks if b.kind == "article"}
        if labels:
            print(
                f"  articles: {len(labels)} headings, {len(with_body)} with body text, "
                f"headings without body={sorted(labels - with_body, key=int)}"
            )
        content = [b for b in blocks if b.kind in ("body", "article", "annex")]
        located = sum(1 for b in content if b.section or b.article)
        print(f"  content blocks with a section/article: {located}/{len(content)}")
        for b in content[len(content) // 2 : len(content) // 2 + 1]:
            print(f"  e.g. [{b.kind}] article={b.article} section={(b.section or '')[:70]!r}")
            print(f"       {b.text[:80]!r}")


if __name__ == "__main__":
    main()
