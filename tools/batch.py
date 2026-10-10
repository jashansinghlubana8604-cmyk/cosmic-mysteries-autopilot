"""Render a run of episodes on this computer and write an upload sheet for scheduling them by hand in YouTube Studio
(used until Google approves the API audit; after that the daily GitHub workflow uploads and schedules by itself).

    python tools/batch.py 001 007          # render 001..007 (skips ones already in out/), then write out/UPLOAD.md
    python tools/batch.py 001 007 --sheet  # only rewrite the upload sheet

Each episode's publish time is its daily slot from config.yaml, counting from schedule.start_date.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos import script  # noqa: E402
from cosmos.pipeline import slot_time  # noqa: E402

OUT = ROOT / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("last")
    ap.add_argument("--sheet", action="store_true", help="only write the upload sheet")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    eps = script.load_all()
    keys = [e.key for e in eps]
    i0 = next(i for i, k in enumerate(keys) if k.startswith(args.first))
    i1 = next(i for i, k in enumerate(keys) if k.startswith(args.last))
    run = eps[i0:i1 + 1]

    if not args.sheet:
        from make import build
        for n, ep in enumerate(run):
            if (OUT / f"{ep.key}.mp4").exists() and (OUT / f"{ep.key}.json").exists():
                print(f"{ep.key}: already rendered", flush=True)
                continue
            nxt = eps[eps.index(ep) + 1] if eps.index(ep) + 1 < len(eps) else None
            teaser = nxt.meta["teaser"] if nxt else "another mystery the universe has not explained"
            t0 = time.time()
            build(ep.path, cfg, OUT, cfg["channel"]["name"], teaser,
                  log_fn=lambda m, k=ep.key: print(f"[{k} {time.time() - t0:5.0f}s] {m}", flush=True))
            print(f"{ep.key}: done in {(time.time() - t0) / 60:.0f} min", flush=True)
            shutil.rmtree(ROOT / "work" / ep.key / "voice", ignore_errors=True)   # keep the disk tidy

    sc = cfg["schedule"]
    tz = ZoneInfo(sc["timezone"])
    start = date.fromisoformat(sc["start_date"])
    lines = ["# Upload sheet", "",
             "For each video: YouTube Studio > Create > Upload videos > pick the .mp4. Paste the title, description and",
             "tags below, upload the thumbnail (.jpg) and the subtitles (.srt, English), set 'No, it's not made for kids',",
             "then Visibility > Schedule and enter the date and time shown (India Standard Time).", ""]
    for ep in run:
        meta_f = OUT / f"{ep.key}.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        day = start + timedelta(days=eps.index(ep))
        hh, mm = slot_time(sc, day)
        when = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
        lines += [f"## {when:%a %d %b %Y, %I:%M %p} IST  ·  {ep.key}", "",
                  f"- Video: `out/{ep.key}.mp4` ({meta['duration'] / 60:.1f} min)",
                  f"- Thumbnail: `out/{ep.key}.jpg`", f"- Subtitles: `out/{ep.key}.srt`", "",
                  "**Title**", "", "```", meta["title"], "```", "",
                  "**Description**", "", "```", meta["description"].strip(), "```", "",
                  "**Tags**", "", "```", ", ".join(meta["tags"]), "```", ""]
    (OUT / "UPLOAD.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'UPLOAD.md'}")


if __name__ == "__main__":
    main()
