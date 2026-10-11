"""Finding footage: search the four CC BY archives (local index), NASA's image and video library and Wikimedia
Commons, score the candidates against a query, and download the winners (cached on disk).

Every asset carries its credit line and licence, so each shot can be credited on screen and in the description.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from .archives import ARCHIVES, INDEX_DIR, LICENSE

log = logging.getLogger("cosmos.sources")

UA = {"User-Agent": "Mozilla/5.0 (CosmicMysteries/1.0; educational documentary; contact via GitHub)"}
_session = requests.Session()
_session.headers.update(UA)
_lock = threading.Lock()

STOP = set("""a an the of in on at to and or for with from by into over under about as is are was were be been this
that these those it its their his her our your my image video photo picture view artist artists artist's impression
illustration animation new first close up closeup""".split())

# talking heads, presenters, events, logos: never usable as space b-roll
BAD_TITLE = re.compile(
    r"hubblecast|esocast|webbcast|chasing starlight|space sparks|podcast|interview|briefing|press conference|"
    r"conference|webinar|panel|ceremony|award|visit|meeting|q&a|live:|science live|this week @nasa|"
    r"this week at nasa|we asked|expert|explains?\b|talks? |speaks|lecture|trailer|teaser|logo|intro|outro|"
    r"\bstaff\b|employees|selfie|portrait of (dr|mr|ms)|signing|students|workshop|exhibit|exhibition|"
    r"museum|planetarium show|fulldome|dome master|360|vr\b|sign language|subtitles|audio description|"
    r"music video|sonification|countdown to|launch coverage|broadcast|newscast|episode \d+|vodcast|"
    r"behind the scenes|making of|timelapse of the construction|construction of|groundbreaking|what's up|"
    r"skywatching|episodio|cosmoview|webcast|eyes on the skies|signature|site visit|inauguration|planetarium|transporter|network's|solar panel|site testing|weather station|halo \(|milky way halo|graphic|infographic|diagram|labell?ed|annotated|\(en|\(es\)|español|espanol|deutsch|français",
    re.I)

BAD_ID = re.compile(r"hubblecast|esocast|webbcast|cs\d{4}|chasing|eyes-skies|hst15|ann\d", re.I)
NASA_GOOD = re.compile(r"artist|illustration|concept|animation|visuali[sz]ation|simulation|hubble|webb|chandra|"
                       r"spitzer|mosaic|nebula|galax|star|planet|comet|asteroid|moon\b|lunar|mars|jupiter|"
                       r"saturn|uranus|neptune|pluto|venus|mercury|\bsun\b|solar|black hole|supernova|earth|"
                       r"universe|cosmic|quasar|pulsar|magnetar|telescope|spacecraft|voyager|probe|orbit|"
                       r"aurora|eclipse|deep field|cluster|milky way|exoplanet|interstellar|meteor|crater", re.I)
NASA_PEOPLE = re.compile(r"astronaut|cosmonaut|\bcrew|specialist|commander|expedition|engineer|technician|"
                         r"employee|student|school|teacher|intern|administrator|director|scientist|researcher|"
                         r"\bdr\.|\bmr\.|\bms\.|portrait|poses|smiles|ceremony|visitors|guests|award|speaks|"
                         r"discusses|press|media day|tour|workers|staff|team members|participants|family",
                         re.I)
NASA_VIDEO_OK = re.compile(r"animation|visuali[sz]ation|simulation|b-roll|broll|time-?lapse|footage|flyover|"
                           r"fly-?through|flight|zoom|artist|concept|tour of|view of|journey|360 view|"
                           r"timelapse|rotating|rotation|spin|orbit", re.I)
NOT_PD = re.compile(r"permission|copyright|©|\(c\)|all rights reserved|courtesy of(?! nasa)", re.I)


@dataclass
class Asset:
    key: str                 # provider:id
    provider: str            # esawebb | esahubble | eso | noirlab | nasa | commons
    kind: str                # video | image
    title: str
    page: str
    credit: str
    license: str
    w: int = 0
    h: int = 0
    score: float = 0.0
    urls: list[str] = field(default_factory=list)
    path: str = ""           # local file once downloaded


def _tokens(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' \-]", " ", s)
    out = []
    for w in re.split(r"[\s\-]+", s):
        w = w.strip("'")
        if not w or w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _match(query_tokens: list[str], title: str) -> float:
    """Fraction of query words found in the title (prefix matches count half)."""
    if not query_tokens:
        return 0.0
    tt = _tokens(title)
    if not tt:
        return 0.0
    hit = 0.0
    for q in query_tokens:
        if q in tt:
            hit += 1
        elif any(t.startswith(q[:5]) for t in tt if len(q) >= 5):
            hit += 0.5
    return hit / len(query_tokens)


# ---- local archive index ----------------------------------------------------------------------------
class _Index:
    rows: dict[tuple[str, str], list[list]] | None = None

    @classmethod
    def load(cls) -> dict[tuple[str, str], list[list]]:
        if cls.rows is None:
            with _lock:
                if cls.rows is None:
                    rows = {}
                    for site in ARCHIVES:
                        for kind in ("videos", "images"):
                            p = INDEX_DIR / f"{site}_{kind}.json"
                            if p.exists():
                                rows[(site, kind)] = json.loads(p.read_text(encoding="utf-8"))
                    cls.rows = rows
        return cls.rows


def search_archives(query: str, kind: str | None) -> list[Asset]:
    qt = _tokens(query)
    out: list[Asset] = []
    for (site, k), rows in _Index.load().items():
        akind = "video" if k == "videos" else "image"
        if kind and kind != akind:
            continue
        a = ARCHIVES[site]
        for row in rows:
            title = row[1]
            if BAD_ID.search(row[0]):
                continue
            m = _match(qt, title)
            if m < 0.6 or BAD_TITLE.search(title):
                continue
            w, h = (row[2], row[3]) if akind == "image" else (1920, 1080)
            if akind == "image" and (min(w, h) < 900 or w / max(1, h) > 3.2 or h / max(1, w) > 2.2):
                continue
            score = m * a["quality"]
            if akind == "video":
                score += 0.18
                if re.search(r"artist|animation|zoom|pan|fly|flight|simulation|visuali", title, re.I):
                    score += 0.08
            if re.search(r"compar|diagram|infographic|chart|label|annotated|graph|spectrum|logo|poster|"
                         r"finder|map of|sky chart|wide-field view|network|location|positions|\bmap\b|grid|"
                         r"mosaic of|scale|sizes|timeline|graphic",
                         title, re.I):
                score -= 0.25
            if re.search(r"potw|potm|iotw", row[0]):
                score -= 0.05
            out.append(Asset(key=f"{site}:{row[0]}", provider=site, kind=akind, title=title,
                             page=a["video_page" if akind == "video" else "image_page"].format(id=row[0]),
                             credit=a["credit"], license=LICENSE, w=w, h=h, score=score))
    return out


# ---- NASA image and video library -----------------------------------------------------------------------
def _cached_json(url: str, params: dict, cache: Path, ttl_days: float = 20) -> dict | None:
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    p = cache / "search" / f"{key}.json"
    if p.exists() and time.time() - p.stat().st_mtime < ttl_days * 86400:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    for attempt in range(3):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data), encoding="utf-8")
                return data
            if r.status_code in (400, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def search_nasa(query: str, kind: str | None, cache: Path) -> list[Asset]:
    media = kind or "image,video"
    by_id = bool(re.fullmatch(r"PIA\d{5}", query.strip()))          # an exact image, e.g. a pinned thumbnail
    params = {"nasa_id": query.strip()} if by_id else {"q": query, "media_type": media, "page_size": 60}
    data = _cached_json("https://images-api.nasa.gov/search", params, cache)
    if not data:
        return []
    qt = _tokens(query)
    out = []
    for item in data.get("collection", {}).get("items", []):
        d = (item.get("data") or [{}])[0]
        title, nid, mt = d.get("title", ""), d.get("nasa_id", ""), d.get("media_type", "")
        if mt not in ("image", "video") or not nid:
            continue
        desc = d.get("description", "")[:600]
        if BAD_TITLE.search(title) or BAD_TITLE.search(desc[:160]):
            continue
        if (not by_id and not NASA_GOOD.search(title)) or NASA_PEOPLE.search(title) or NASA_PEOPLE.search(desc[:250]):
            continue
        if re.match(r"^[A-Z .,\-]{12,}", title):           # "KENNEDY SPACE CENTER, FLA. - ..." event captions
            continue
        if mt == "video" and not NASA_VIDEO_OK.search(title):
            continue
        sec = d.get("secondary_creator") or d.get("photographer") or ""
        if NOT_PD.search(sec) or NOT_PD.search(desc) or NOT_PD.search(d.get("copyright", "") or ""):
            continue
        m = 1.0 if by_id else _match(qt, title)
        if m < 0.66:
            continue
        score = m * 0.9 + (0.15 if mt == "video" else 0.0)
        credit = "NASA" + (f"/{d['center']}" if d.get("center") else "")
        if sec and len(sec) < 80:
            credit = f"{credit} ({sec})"
        out.append(Asset(key=f"nasa:{nid}", provider="nasa", kind=mt, title=title,
                         page=f"https://images.nasa.gov/details/{nid}", credit=credit, license="Public domain (NASA)",
                         score=score))
    return out


# ---- Wikimedia Commons ------------------------------------------------------------------------------------
OK_LICENSES = re.compile(r"^(public domain|pd|cc0|cc by(-sa)? ?\d|cc-by(-sa)?-\d)", re.I)


def search_commons(query: str, cache: Path) -> list[Asset]:
    data = _cached_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": 25, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 2560}, cache)
    if not data:
        return []
    qt = _tokens(query)
    out = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = html.unescape(re.sub("<[^>]+>", "", meta.get("LicenseShortName", {}).get("value", "")))
        if not OK_LICENSES.match(lic.strip()):
            continue
        w, h = int(info.get("width", 0)), int(info.get("height", 0))
        if min(w, h) < 700 or w / max(1, h) > 3.2 or h / max(1, w) > 2.0:
            continue
        title = page.get("title", "").removeprefix("File:").rsplit(".", 1)[0].replace("_", " ")
        m = _match(qt, title)
        if m < 0.75 or BAD_TITLE.search(title):
            continue
        if re.search(r"linocut|print|painting|drawing|sketch|poster|stamp|coin|logo|cartoon|comic|meme|shirt|"
                     r"tattoo|mural|graffiti|book cover|album|screenshot|montage|collage|sign\b|plaque", title, re.I):
            continue
        artist = html.unescape(re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", ""))).strip()
        artist = re.sub(r"\s+", " ", artist)[:60]
        if not artist or len(artist) < 3:
            artist = "Wikimedia Commons"
        credit = f"{artist or 'Wikimedia Commons'} / {lic}"
        url = info.get("thumburl") or info.get("url")
        out.append(Asset(key=f"commons:{page.get('pageid')}", provider="commons", kind="image", title=title,
                         page=info.get("descriptionurl", ""), credit=credit, license=lic, w=w, h=h,
                         score=m * 0.8, urls=[url]))
    return out


# ---- query dispatch ---------------------------------------------------------------------------------------
def search(query: str, cache: Path) -> list[Asset]:
    """`query` may carry a prefix: v: (video only) i: (image only) c: (Commons) n: (NASA)."""
    q = query.strip()
    prefix = ""
    m = re.match(r"^([vicn]):\s*(.+)$", q)
    if m:
        prefix, q = m.group(1), m.group(2)
    kind = {"v": "video", "i": "image"}.get(prefix)
    found: list[Asset] = []
    if prefix == "c":
        found += search_commons(q, cache)
        found += [a for a in search_nasa(q, "image", cache)]
    elif prefix == "n":
        found += search_nasa(q, kind, cache)
    else:
        found += search_archives(q, kind)
        found += search_nasa(q, kind, cache)
    found.sort(key=lambda a: -a.score)
    return found


# ---- downloads ----------------------------------------------------------------------------------------------
def _page_html(url: str, cache: Path) -> str:
    key = hashlib.md5(url.encode()).hexdigest()
    p = cache / "pages" / f"{key}.html"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=40)
            if r.status_code == 200:
                try:
                    text = r.content.decode("utf-8")
                except UnicodeDecodeError:
                    text = r.content.decode("cp1252", errors="replace")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
                return text
            if r.status_code == 404:
                return ""
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return ""


def _archive_credit(page: str, default: str) -> str:
    m = re.search(r"Credit:?\s*</[^>]+>(.{0,600}?)</(?:div|p|td)>", page, re.S)
    if m:
        c = re.sub(r"<[^>]+>", "", m.group(1))
        c = re.sub(r"\s+", " ", html.unescape(c)).strip(" .:")
        c = re.sub(r"\(\s*\)", "", c).strip()
        if c.count("(") > c.count(")"):
            c = c.rsplit("(", 1)[0].strip(" ,")
        if 2 < len(c) < 200:
            return c
    return default


def resolve_urls(a: Asset, cache: Path) -> list[str]:
    """Download candidates for an asset, best first."""
    if a.urls:
        return a.urls
    if a.provider in ARCHIVES:
        arch = ARCHIVES[a.provider]
        page = _page_html(a.page, cache)
        if page:
            a.credit = _archive_credit(page, a.credit)
        if a.kind == "image":
            urls = [arch["image_file"].format(id=a.key.split(":", 1)[1])]
            if page:
                urls = re.findall(r'https?://[^"\']+/archives/images/(?:large|publicationjpg|screen)/[^"\']+\.jpg', page)
                urls.sort(key=lambda u: ("/large/" not in u, "/publicationjpg/" not in u))
                urls = urls or [arch["image_file"].format(id=a.key.split(":", 1)[1])]
            a.urls = urls
        else:
            vids = re.findall(r'https?://[^"\']+\.(?:mp4|m4v|mov)', page or "")
            pref = ["hd_1080p25_screen", "hd_1080_screen", "hd_and_apple", "ultra_hd/", "large_qt", "broadcast",
                    "medium_podcast"]

            def rank(u: str) -> int:
                for i, p in enumerate(pref):
                    if p in u:
                        return i
                return 99
            vids = sorted(set(v for v in vids if "h265" not in v and "fulldome" not in v and "dome" not in v),
                          key=rank)
            a.urls = [v for v in vids if rank(v) < 99]
        return a.urls
    if a.provider == "nasa":
        nid = a.key.split(":", 1)[1]
        data = _cached_json(f"https://images-api.nasa.gov/asset/{requests.utils.quote(nid)}", {}, cache, 90)
        hrefs = [i.get("href", "") for i in (data or {}).get("collection", {}).get("items", [])]
        hrefs = [h.replace("http://", "https://").replace(" ", "%20") for h in hrefs]
        if a.kind == "image":
            order = ["~large.jpg", "~orig.jpg", "~orig.png", "~medium.jpg"]
        else:
            order = ["~orig.mp4", "~large.mp4", "~medium.mp4", "~mobile.mp4"]
        a.urls = [h for o in order for h in hrefs if h.lower().endswith(o)]
        return a.urls
    return a.urls


def download(a: Asset, cache: Path, max_mb: int = 400) -> Path | None:
    """Fetch the asset's file (cached). Returns None when nothing usable could be downloaded."""
    media = cache / "media"
    media.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", a.key)[:80]
    for ext in (".mp4", ".mov", ".m4v", ".jpg", ".png"):
        p = media / f"{stem}{ext}"
        if p.exists() and p.stat().st_size > 20_000:
            a.path = str(p)
            return p
    for url in resolve_urls(a, cache)[:3]:
        ext = "." + url.rsplit(".", 1)[-1].lower().split("?")[0]
        if ext not in (".mp4", ".mov", ".m4v", ".jpg", ".jpeg", ".png"):
            ext = ".jpg" if a.kind == "image" else ".mp4"
        ext = ".jpg" if ext == ".jpeg" else ext
        dst = media / f"{stem}{ext}"
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            with _session.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    continue
                size = int(r.headers.get("content-length") or 0)
                if a.kind == "image" and size > 150_000_000:     # gigapixel originals: skip
                    log.info("skip %s (%.0f MB image)", url, size / 1e6)
                    continue
                if size > max_mb * 1_000_000:
                    log.info("skip %s (%.0f MB)", url, size / 1e6)
                    continue
                got = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if got > max_mb * 1_000_000:
                            raise IOError("too large")
            if tmp.stat().st_size < 20_000:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dst)
            a.path = str(dst)
            return dst
        except (requests.RequestException, IOError) as e:
            log.info("download failed %s: %s", url, e)
            tmp.unlink(missing_ok=True)
    return None


def asset_to_dict(a: Asset) -> dict:
    return asdict(a)
