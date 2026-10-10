"""Render one episode locally (no upload).

    python make.py 001                 # episode whose file starts with 001
    python make.py wow-signal
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cosmos import edit, metadata, script, thumbnail  # noqa: E402


def find(key: str) -> Path:
    hits = [p for p in sorted(script.EPISODES_DIR.glob("[0-9]*.md")) if p.stem.startswith(key) or key in p.stem]
    if not hits:
        raise SystemExit(f"no episode matches {key!r}")
    return hits[0]


def build(path: Path, cfg: dict, out_dir: Path, channel: str, next_teaser: str, seed: int | None = None,
          log_fn=print) -> dict:
    ep = script.parse(path)
    seed = seed if seed is not None else sum(map(ord, ep.key))
    work = ROOT / "work" / ep.key
    cache = ROOT / "work" / "cache"
    res = edit.produce(ep, cfg, work, cache, seed=seed, channel=channel, next_teaser=next_teaser, log_fn=log_fn)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{ep.key}.mp4"
    shutil.move(str(res.video), video)
    best = thumbnail.pick(ep, cache) or thumbnail.striking(res.stills) or res.hero
    thumb = None
    if best:
        thumb = thumbnail.make(best, ep.meta["thumb"], out_dir / f"{ep.key}.jpg", ep.meta.get("badge", ""), ep.meta.get("thumb_style", "cinematic"),
                       ep.meta.get("thumb_mark", ""))
    shutil.copy(res.srt, out_dir / f"{ep.key}.srt")
    meta = {
        "key": ep.key, "title": ep.title,
        "description": metadata.description(ep, res.chapters, res.credits, res.music, channel),
        "tags": metadata.tags(ep), "duration": res.duration,
        "chapters": res.chapters, "video": str(video), "thumbnail": str(thumb) if thumb else "",
        "captions": str(out_dir / f"{ep.key}.srt"),
    }
    (out_dir / f"{ep.key}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--next", default="another mystery the universe has not explained")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    t0 = time.time()
    meta = build(find(args.episode), cfg, ROOT / "out", args.channel or cfg["channel"]["name"], args.next,
                 log_fn=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True))
    print(json.dumps({k: v for k, v in meta.items() if k != "description"}, indent=1))
    print(meta["description"])


if __name__ == "__main__":
    main()
