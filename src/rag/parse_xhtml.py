"""XHTML parser (step 2.3): EUR-Lex / CELLAR XHTML (CRR) -> list[Block].

Unlike PDF, the XHTML already carries the legal structure in its markup:
  <div id="rct_12">       recital 12
  <div id="art_92">       Article 92   (<p class="oj-ti-art"> + <p class="oj-sti-art"> title)
  <div id="prt_III.tis_I.cpt_2">   Part / Title / Chapter containers
  <div id="anx_I">        Annex I
  2-column <table> (4% / 96%)       an enumeration point "(a) ..."
  <table class="oj-table">          a real data table
  <p class="oj-note">               a footnote
So there is no header/footer cleaning and no guessing: we read structure, we do not infer it.

Debug run: uv run python -m rag.parse_xhtml
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from rag.config import PROJECT_ROOT, load_config
from rag.schema import Block, Kind

NS = "{http://www.w3.org/1999/xhtml}"
ARTICLE_ID = re.compile(r"^art_(\w+)$")
CONTAINER_ID = re.compile(r"(^|\.)(prt|tis|cpt|sct)_\w+$")  # Part, Title, Chapter, Section


def tag(el: etree._Element) -> str:
    """Tag name without the XHTML namespace: '{http://...}p' -> 'p'."""
    return el.tag.replace(NS, "") if isinstance(el.tag, str) else ""


def classes(el: etree._Element) -> set[str]:
    return set((el.get("class") or "").split())


def text_of(el: etree._Element) -> str:
    """All text inside the element, whitespace collapsed."""
    return " ".join("".join(el.itertext()).split())


class XhtmlParser:
    def __init__(self, path: Path):
        parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
        self.root = etree.parse(str(path), parser).getroot()
        self.doc_id = path.stem
        self.blocks: list[Block] = []
        self.stats = {"tables": 0}
        # Cache key = the element's id ATTRIBUTE. Never Python's id(el): lxml creates element
        # proxies on the fly, so id() values get reused and the cache returns wrong headings.
        self._heading_cache: dict[str, str] = {}

    # ----- context: where are we in the legal structure? -----

    def context(self, el: etree._Element) -> tuple[Kind, str | None, str | None]:
        """Return (kind, article, section) from the element's ancestors' ids."""
        kind: Kind = "body"
        article = None
        path: list[str] = []
        for anc in el.iterancestors():
            anc_id = anc.get("id") or ""
            m = ARTICLE_ID.match(anc_id)
            if m and article is None:
                article = m.group(1)
            if CONTAINER_ID.search(anc_id):
                path.append(self.container_heading(anc))
            if anc_id.startswith("rct_"):
                kind = "recital"
            elif anc_id.startswith("anx_"):
                kind = "annex"
                path.append(self.container_heading(anc))
        if article is not None:
            kind = "article"
        section = " > ".join(reversed(path)) or None
        return kind, article, section

    def container_heading(self, div: etree._Element) -> str:
        """'PART THREE CAPITAL REQUIREMENTS' = section-1 label + its eli-title text."""
        key = div.get("id") or ""
        if key not in self._heading_cache:
            parts = [text_of(c) for c in div if tag(c) == "p"][:1]  # e.g. 'PART THREE'
            parts += [text_of(c) for c in div if "eli-title" in classes(c)][:1]
            self._heading_cache[key] = " ".join(parts) or key
        return self._heading_cache[key]

    # ----- emitting blocks -----

    def emit(self, text: str, el: etree._Element, kind: Kind | None = None) -> None:
        # Collapse spaces inside each line but keep line breaks (table rows)
        text = "\n".join(" ".join(ln.split()) for ln in text.splitlines() if ln.strip())
        if not text:
            return
        ctx_kind, article, section = self.context(el)
        self.blocks.append(
            Block(
                doc_id=self.doc_id,
                order=len(self.blocks),
                text=text,
                kind=kind or ctx_kind,
                page=None,  # XHTML has no pages (decision 2.1, option A)
                article=article,
                section=section,
            )
        )

    # ----- walking the tree in document order -----

    def walk(self, el: etree._Element, prefix: str = "") -> None:
        for child in el:
            name, cls = tag(child), classes(child)
            if name in {"figure", "img"}:
                continue  # formulas are JPEG images without text (counted in parse())
            elif name == "table" and "oj-hd-date" not in etree.tostring(child, encoding=str):
                self.table(child, prefix)
                prefix = ""
            elif name == "p":
                prefix = self.paragraph(child, cls, prefix)
            elif name == "div" and "eli-title" not in cls:
                self.walk(child, prefix)
                prefix = ""
            elif name not in {"div", "table", "hr", "col"}:
                self.walk(child, prefix)

    def paragraph(self, p: etree._Element, cls: set[str], prefix: str) -> str:
        """Emit one <p>; return the prefix still waiting to be used (if any)."""
        if cls & {"oj-ti-art", "oj-ti-section-1"}:
            # Heading = label + the title in the next sibling <div class="eli-title">
            nxt = p.getnext()
            title = text_of(nxt) if nxt is not None and "eli-title" in classes(nxt) else ""
            self.emit(f"{text_of(p)} {title}", p, kind="title")
            return prefix
        if "oj-note" in cls:
            self.emit(text_of(p), p, kind="footnote")
            return prefix
        if "oj-doc-ti" in cls:
            self.emit(text_of(p), p, kind="title")
            return prefix
        text = text_of(p)
        if not text:  # e.g. <p><figure><img/></figure></p>: a formula image, no text
            return prefix
        self.emit(f"{prefix} {text}", p)
        return ""

    def table(self, table: etree._Element, prefix: str) -> None:
        if "oj-table" in classes(table):
            # Real data table: one Block, rows on separate lines, cells joined by " | "
            self.stats["tables"] += 1
            rows = []
            for tr in table.iter(f"{NS}tr"):
                cells = [text_of(td) for td in tr if tag(td) == "td"]
                rows.append(" | ".join(cells))
            self.emit("\n".join(rows), table, kind="table")
            return
        # Layout table = enumeration: first cell is the point label "(a)", the rest is content
        for tr in table.iterchildren(f"{NS}tbody", f"{NS}tr"):
            rows = [tr] if tag(tr) == "tr" else list(tr.iterchildren(f"{NS}tr"))
            for row in rows:
                cells = [td for td in row if tag(td) == "td"]
                if len(cells) < 2:
                    for td in cells:
                        self.walk(td, prefix)
                    continue
                label = text_of(cells[0])
                for td in cells[1:]:
                    self.walk(td, f"{prefix} {label}".strip())

    def parse(self) -> tuple[list[Block], dict]:
        body = self.root.find(f"{NS}body")
        self.walk(body)
        self.stats["formulas_dropped"] = sum(1 for _ in body.iter(f"{NS}img"))
        self.stats["blocks"] = len(self.blocks)
        self.stats["articles"] = len({b.article for b in self.blocks if b.article})
        self.stats["recital_blocks"] = sum(b.kind == "recital" for b in self.blocks)
        return self.blocks, self.stats


def parse_xhtml(path: Path) -> tuple[list[Block], dict]:
    return XhtmlParser(path).parse()


def main() -> None:
    """Debug: stats, blocks per kind, and samples (Article 92, a recital, a table)."""
    cfg = load_config()
    for path in sorted((PROJECT_ROOT / cfg["paths"]["raw_dir"]).glob("*.xhtml")):
        blocks, s = parse_xhtml(path)
        print(f"\n=== {path.stem} === {s}")
        kinds: dict[str, int] = {}
        for b in blocks:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        print(f"  blocks per kind: {kinds}")
        art92 = [b for b in blocks if b.article == "92"]
        print(f"  Article 92 -> {len(art92)} blocks, section = {art92[0].section!r}")
        for b in art92[:5]:
            print(f"    [{b.kind:7}] {b.text[:100]!r}")
        samples = [next(b for b in blocks if b.kind == k) for k in ("recital", "table")]
        for b in samples:
            print(f"  first {b.kind}: {b.text[:100]!r}  (section={b.section!r})")


if __name__ == "__main__":
    main()
