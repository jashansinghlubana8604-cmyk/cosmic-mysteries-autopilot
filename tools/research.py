"""Proof-of-concept research: which space-mystery videos actually won on YouTube?

Searches the niche, pulls stats for every result, and ranks by views, views per day, and "outlier score"
(views relative to the channel's subscriber count = the topic/packaging did the work, not the channel size).

    python tools/research.py  -> research/videos.json, research/report.md, research/thumbs/*.jpg
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research"

QUERIES = [
    "space mystery", "unexplained space mysteries", "mysteries of the universe", "scientists can't explain space",
    "strange signal from space", "something strange in space", "black hole documentary", "edge of the universe",
    "james webb telescope discovery", "interstellar object", "voyager spacecraft", "end of the universe",
    "biggest things in the universe", "dark matter mystery", "terrifying space", "space documentary sleep",
    "what is outside the universe", "alien megastructure star", "most mysterious planet", "deep space mystery",
]


def token(cred_path: Path) -> str:
    c = json.loads(cred_path.read_text())
    data = urllib.parse.urlencode({"client_id": c["client_id"], "client_secret": c["client_secret"],
                                   "refresh_token": c["refresh_token"], "grant_type": "refresh_token"}).encode()
    return json.load(urllib.request.urlopen("https://oauth2.googleapis.com/token", data))["access_token"]


def api(tok: str, path: str, **params) -> dict:
    url = f"https://www.googleapis.com/youtube/v3/{path}?{urllib.parse.urlencode(params)}"
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})))


def iso_dur(s: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    h, mi, se = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + se


def main() -> None:
    cred = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Jashan\Desktop\stickman_tool\channel\.secrets\youtube.json")
    tok = token(cred)
    OUT.mkdir(exist_ok=True)
    ids: dict[str, str] = {}
    for q in QUERIES:
        for dur in ("medium", "long"):
            r = api(tok, "search", part="id", q=q, type="video", order="viewCount", maxResults=50,
                    publishedAfter="2023-06-01T00:00:00Z", relevanceLanguage="en", videoDuration=dur)
            for it in r.get("items", []):
                ids.setdefault(it["id"]["videoId"], q)
    vids = []
    idl = list(ids)
    for i in range(0, len(idl), 50):
        r = api(tok, "videos", part="snippet,statistics,contentDetails", id=",".join(idl[i:i + 50]))
        vids += r.get("items", [])
    chans = {}
    cl = list({v["snippet"]["channelId"] for v in vids})
    for i in range(0, len(cl), 50):
        r = api(tok, "channels", part="statistics,snippet", id=",".join(cl[i:i + 50]))
        for c in r.get("items", []):
            chans[c["id"]] = c
    now = datetime.now(timezone.utc)
    rows = []
    for v in vids:
        s, st = v["snippet"], v.get("statistics", {})
        views = int(st.get("viewCount", 0))
        dur = iso_dur(v["contentDetails"].get("duration"))
        if dur < 360 or views < 20000:
            continue
        age = max(1, (now - datetime.fromisoformat(s["publishedAt"].replace("Z", "+00:00"))).days)
        ch = chans.get(s["channelId"], {})
        subs = int(ch.get("statistics", {}).get("subscriberCount", 0) or 0)
        rows.append({"id": v["id"], "title": s["title"], "channel": s["channelTitle"], "subs": subs,
                     "views": views, "per_day": views / age, "outlier": views / max(subs, 1000),
                     "minutes": round(dur / 60, 1), "published": s["publishedAt"][:10], "query": ids[v["id"]],
                     "thumb": (s["thumbnails"].get("maxres") or s["thumbnails"].get("high"))["url"]})
    rows.sort(key=lambda r: -r["views"])
    (OUT / "videos.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    lines = ["# Space-mystery proof of concept", "", f"{len(rows)} videos (6+ min, since June 2023)", "",
             "## Top by outlier score (views / subscribers)", ""]
    for r in sorted(rows, key=lambda r: -r["outlier"])[:60]:
        lines.append(f"- {r['outlier']:.1f}x | {r['views']:,} views | {r['minutes']}m | {r['channel']} "
                     f"({r['subs']:,}) | {r['title']}")
    lines += ["", "## Top by views per day", ""]
    for r in sorted(rows, key=lambda r: -r["per_day"])[:60]:
        lines.append(f"- {r['per_day']:,.0f}/day | {r['views']:,} | {r['minutes']}m | {r['channel']} | {r['title']}")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    tdir = OUT / "thumbs"
    tdir.mkdir(exist_ok=True)
    top = sorted(rows, key=lambda r: -(r["outlier"] * 0.5 + r["per_day"] / 2000))[:48]
    for k, r in enumerate(top):
        try:
            urllib.request.urlretrieve(r["thumb"], tdir / f"{k:02d}_{r['id']}.jpg")
        except Exception:  # noqa: BLE001
            pass
    print(len(rows), "videos")


if __name__ == "__main__":
    main()
