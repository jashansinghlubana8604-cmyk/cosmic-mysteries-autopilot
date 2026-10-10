"""Build the local search index of the ESA/Hubble, ESA/Webb, ESO and NOIRLab media archives.

All four archives publish their images and videos under CC BY 4.0. Their own search is uneven, so we crawl the
listing pages once (titles, ids, image sizes) and search them ourselves. Run again now and then to pick up new
releases; the daily workflow also refreshes it weekly.

    python tools/build_index.py            # all archives
    python tools/build_index.py esawebb    # one archive
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos.archives import ARCHIVES, INDEX_DIR  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (cosmic-mysteries index; educational use)"}
SESSION = requests.Session()
SESSION.headers.update(UA)


def fetch(url: str) -> str:
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=40)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return r.content.decode("cp1252", errors="replace")
        except requests.RequestException:
            time.sleep(2 + 3 * attempt)
    return ""


def total_items(page: str) -> int:
    m = re.search(r"Showing\s+\d+\s+to\s+\d+\s+of\s+([\d,]+)", page)
    return int(m.group(1).replace(",", "")) if m else 0


def parse_images(page: str) -> list[list]:
    out = []
    block = re.search(r"var images = \[(.*?)\];", page, re.S)
    if not block:
        return out
    for m in re.finditer(r"id:\s*'([^']+)',\s*title:\s*'((?:[^'\\]|\\.)*)',\s*width:\s*(\d+),\s*height:\s*(\d+)",
                         block.group(1)):
        title = html.unescape(m.group(2).replace("\\'", "'").replace('\\"', '"'))
        out.append([m.group(1), title, int(m.group(3)), int(m.group(4))])
    return out


def parse_videos(page: str) -> list[list]:
    found: dict[str, str] = {}
    for m in re.finditer(r'data-title="([^"]*)"[^>]*?data-url="(?:/public)?/videos/([a-z0-9\-_]+)/"', page):
        found.setdefault(m.group(2), html.unescape(m.group(1)))
    for m in re.finditer(r'href="(?:/public)?/videos/([a-z0-9\-_]+)/"[^>]*>\s*<span class="title">([^<]+)<', page):
        found.setdefault(m.group(1), html.unescape(m.group(2)))
    for m in re.finditer(r'href="(?:/public)?/videos/([a-z0-9\-_]+)/".{0,600}?alt="(?:Video thumbnail for )?([^"]+)"',
                         page, re.S):
        found.setdefault(m.group(1), html.unescape(m.group(2)))
    skip = {"search", "viewall", "list", "page", "archive", "feed", "potw", "potm", "iotw", "comparisons"}
    return [[k, v.strip()] for k, v in found.items() if k not in skip and v.strip()]


def crawl(site: str, kind: str) -> list[list]:
    a = ARCHIVES[site]
    listing = a[f"{kind}_list"]
    first = fetch(listing.format(page=1))
    n = total_items(first)
    pages = max(1, -(-n // 50))
    parse = parse_images if kind == "images" else parse_videos
    rows = {r[0]: r for r in parse(first)}

    def one(p: int) -> list[list]:
        return parse(fetch(listing.format(page=p)))

    with ThreadPoolExecutor(4) as ex:
        for got in ex.map(one, range(2, pages + 1)):
            for r in got:
                rows.setdefault(r[0], r)
    print(f"{site:9s} {kind:6s} {len(rows):6d} of {n} (pages {pages})", flush=True)
    return list(rows.values())


def main(sites: list[str]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for site in sites:
        for kind in ("videos", "images"):
            rows = crawl(site, kind)
            if len(rows) < 20:
                print(f"  ! {site} {kind}: too few rows, keeping the previous index", flush=True)
                continue
            path = INDEX_DIR / f"{site}_{kind}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(ARCHIVES))
