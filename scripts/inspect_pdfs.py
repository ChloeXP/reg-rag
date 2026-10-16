"""Step 2.0 - inspect raw PDFs before writing the parser.

For each PDF, print: page count, characters per page, near-empty pages (scanned?),
number of "Article N" headings, the most repeated text in the top/bottom margins
(header/footer candidates), and a snippet of the cover page (to check the date).

Run: uv run python scripts/inspect_pdfs.py
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf

from rag.config import PROJECT_ROOT, load_config

MARGIN = 0.08  # top/bottom 8% of page height = header/footer zone
ARTICLE_RE = re.compile(r"^\s*Article\s+(\d+|premier)\b", re.MULTILINE)


def normalize(text: str) -> str:
    """Collapse whitespace and mask digits, so 'Page 3' and 'Page 4' count as one pattern."""
    return re.sub(r"\d+", "#", " ".join(text.split()))[:70]


def inspect(path: Path) -> None:
    top, bottom = Counter(), Counter()
    chars, pages_text = [], []

    with pymupdf.open(path) as doc:
        n = doc.page_count
        for page in doc:
            h = page.rect.height
            text = page.get_text("text")
            pages_text.append(text)
            chars.append(len(text.strip()))
            # block = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text, 1 = image
            for _x0, y0, _x1, y1, btext, _no, btype in page.get_text("blocks"):
                if btype != 0 or not btext.strip():
                    continue
                if y1 < h * MARGIN:
                    top[normalize(btext)] += 1
                elif y0 > h * (1 - MARGIN):
                    bottom[normalize(btext)] += 1

    near_empty = sum(c < 50 for c in chars)
    n_articles = len(ARTICLE_RE.findall("\n".join(pages_text)))
    cover = " ".join(pages_text[0].split())[:200]
    median = sorted(chars)[n // 2]

    print(f"\n=== {path.name} ===")
    print(f"pages={n}  chars/page min={min(chars)} median={median} max={max(chars)}")
    print(f"near_empty_pages={near_empty}  'Article N' headings={n_articles}")
    for label, counter in (("TOP", top), ("BOTTOM", bottom)):
        for pattern, count in counter.most_common(3):
            print(f"  {label:6} {count / n:4.0%}  {pattern!r}")
    print(f"  COVER  {cover!r}")


def main() -> None:
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    pdfs = sorted(raw_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {raw_dir}")
    for path in pdfs:
        inspect(path)


if __name__ == "__main__":
    main()
