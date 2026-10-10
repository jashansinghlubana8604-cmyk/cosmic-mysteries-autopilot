"""Preview thumbnails for episodes without rendering the videos.

    python tools/thumbs.py            # every episode -> out/thumbs/
    python tools/thumbs.py 003        # one episode
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos import script, thumbnail  # noqa: E402
from cosmos.edit import AWE, EARTHLY, Footage  # noqa: E402


def candidates(ep, fx):
    return thumbnail.candidates(ep, fx)


def main() -> None:
    keys = sys.argv[1:]
    fx = Footage(ROOT / "work" / "cache", 1)
    out = ROOT / "out" / "thumbs"
    for p in sorted(script.EPISODES_DIR.glob("[0-9]*.md")):
        if keys and not any(p.stem.startswith(k) for k in keys):
            continue
        ep = script.parse(p)
        best = thumbnail.pick(ep, ROOT / "work" / "cache")
        if not best:
            print(f"{ep.key}: no image found")
            continue
        thumbnail.make(best, ep.meta["thumb"], out / f"{ep.key}.jpg", ep.meta.get("badge", ""), ep.meta.get("thumb_style", "cinematic"),
                       ep.meta.get("thumb_mark", ""))
        print(f"{ep.key}: {best.name}")


if __name__ == "__main__":
    main()
