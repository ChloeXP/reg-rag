"""Download source documents listed in data/sources.yaml and write data/manifest.csv.

Acquisition methods (field `method` in sources.yaml):
    http    direct download of `url` (default)
    cellar  EU Publications Office API by CELEX id (the EUR-Lex website blocks scripts)
    manual  file saved by hand from a browser; the script only verifies and registers it

File formats (field `format`): pdf (default) | xhtml

Usage:
    uv run python -m rag.download          # fetch missing files only (idempotent)
    uv run python -m rag.download --force  # re-fetch everything and detect upstream changes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from rag.config import PROJECT_ROOT, load_config

USER_AGENT = "Mozilla/5.0 (Macintosh) reg-rag/0.1 (open-source portfolio project)"
CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
LANG_ISO3 = {"fr": "fra", "en": "eng"}  # CELLAR expects 3-letter language codes
FORMATS = {
    "pdf": {"mime": "application/pdf", "ext": ".pdf"},
    "xhtml": {"mime": "application/xhtml+xml", "ext": ".xhtml"},
}
OK_STATUSES = ("ok", "cached", "manual")

SOURCE_FIELDS = ["doc_id", "work_id", "title", "publisher", "doc_type", "language", "published"]
MANIFEST_FIELDS = SOURCE_FIELDS + [
    "method",
    "format",
    "url",
    "final_url",
    "filename",
    "size_bytes",
    "sha256",
    "downloaded_at",
    "status",
    "notes",
]


def sha256_of(path: Path) -> str:
    """Content fingerprint: any change in the file changes the hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_previous_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def looks_valid(content: bytes, fmt: str) -> bool:
    """Check the bytes really are the expected format (a 200 status is not enough)."""
    if fmt == "pdf":
        return content.startswith(b"%PDF")
    return b"<html" in content[:4096].lower()


def download_file(
    client: httpx.Client, url: str, dest: Path, fmt: str, headers: dict | None = None
) -> tuple[str, str]:
    """Download url to dest. Returns (status, final_url). Never leaves a partial or invalid file."""
    try:
        resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return f"error({type(exc).__name__})", url
    if resp.status_code != 200:
        return f"http_{resp.status_code}", str(resp.url)
    if not looks_valid(resp.content, fmt):  # e.g. an HTML anti-bot page served with 200
        ctype = resp.headers.get("content-type", "?").split(";")[0]
        return f"invalid_{fmt}({ctype})", str(resp.url)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)  # atomic rename
    return "ok", str(resp.url)


def fetch(client: httpx.Client, src: dict, dest: Path) -> tuple[str, str]:
    """Dispatch on the acquisition method declared in sources.yaml."""
    method, fmt = src.get("method", "http"), src.get("format", "pdf")
    if method == "manual":
        return ("manual" if dest.exists() else "manual_missing"), src["url"]
    if method == "cellar":
        url = CELLAR_URL.format(celex=src["celex"])
        headers = {"Accept": FORMATS[fmt]["mime"], "Accept-Language": LANG_ISO3[src["language"]]}
        return download_file(client, url, dest, fmt, headers)
    return download_file(client, src["url"], dest, fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download regulatory source documents.")
    parser.add_argument("--force", action="store_true", help="re-fetch even if file exists")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    manifest_path = PROJECT_ROOT / cfg["paths"]["manifest"]
    sources_path = PROJECT_ROOT / cfg["paths"].get("sources", "data/sources.yaml")
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))["sources"]
    previous = load_previous_manifest(manifest_path)
    rows, missing_manual, n_failed = [], [], 0

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120
    ) as client:
        for src in sources:
            doc_id = src["doc_id"]
            method, fmt = src.get("method", "http"), src.get("format", "pdf")
            dest = raw_dir / f"{doc_id}{FORMATS[fmt]['ext']}"
            prev = previous.get(doc_id, {})

            if dest.exists() and not args.force and method != "manual":
                status = "cached"
                final_url = prev.get("final_url", src["url"])
                downloaded_at = prev.get("downloaded_at", "")
            else:
                status, final_url = fetch(client, src, dest)
                if method == "manual":
                    mtime = dest.stat().st_mtime if dest.exists() else None
                    downloaded_at = (
                        datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds")
                        if mtime
                        else ""
                    )
                else:
                    downloaded_at = datetime.now(UTC).isoformat(timespec="seconds")
                    time.sleep(1)  # be polite to public servers

            ok = dest.exists() and status in OK_STATUSES
            sha = sha256_of(dest) if ok else ""
            if ok and prev.get("sha256") and prev["sha256"] != sha:
                status = "changed"  # same source, different bytes: document updated upstream
            n_failed += not ok
            if status == "manual_missing":
                missing_manual.append((src["url"], dest))
            size = dest.stat().st_size if ok else 0

            rows.append(
                {k: src.get(k, "") for k in SOURCE_FIELDS}
                | {
                    "method": method,
                    "format": fmt,
                    "url": src["url"],
                    "final_url": final_url,
                    "filename": dest.name if ok else "",
                    "size_bytes": size or "",
                    "sha256": sha,
                    "downloaded_at": downloaded_at,
                    "status": status,
                    "notes": src.get("notes", ""),
                }
            )
            print(f"{status:24} {method:7} {fmt:6} {doc_id:32} {size / 1e6:6.2f} MB")

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows) - n_failed}/{len(rows)} documents OK -> {manifest_path.name}")
    for url, dest in missing_manual:
        print(f"\nManual download needed:\n  open    {url}\n  save as {dest}")
    return 1 if n_failed else 0  # non-zero exit code lets CI detect failures


if __name__ == "__main__":
    sys.exit(main())
