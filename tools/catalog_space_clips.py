"""Catalog the local Storyblocks space clips (D:/Jashan/Desktop/Space/Downloaded Videos) with ffprobe only (light on RAM).

Writes assets/space_clips.yaml: one entry per usable clip (file, title, seconds, size, keywords) and a list of
rejected ones with the reason. Gates (Luxury Lane style): native 16:9 landscape, >=1280 wide, >=4 s, not 360/VR,
not a loop-only/vertical clip, no presenters/people/logos/text titles.
Usage: python tools/catalog_space_clips.py [source_dir]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Jashan\Desktop\Space\Downloaded Videos")
OUT = ROOT / "assets" / "space_clips.yaml"
BAD = re.compile(r"360|\bvr\b|virtual reality|logo|title|text|typography|lower third|countdown|green screen|chroma|"
                 r"alpha|overlay|transition|particles? background|loop(?:ing)? background|hud\b|interface|"
                 r"business|woman|man |people|person|child|kid|girl|boy|family|couple|astronaut|cosmonaut|"
                 r"selfie|portrait|face|hand|finger|smartphone|laptop|office|cartoon|clipart|sticker", re.I)
STOP = set("a an the of in on at to and or for with from by into over under about as is are was were this that "
           "4k 10 bit hd uhd video footage animation 3d render rendering loop shot view".split())


def probe(p: Path) -> dict | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(p)],
                       capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]
        n, d = (s.get("r_frame_rate") or "0/1").split("/")
        return {"w": int(s["width"]), "h": int(s["height"]), "fps": round(float(n) / max(float(d), 1), 2),
                "seconds": round(float(j["format"]["duration"]), 2)}
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ok, bad = [], []
    for p in sorted(SRC.glob("*.mp4")):
        title = p.stem
        info = probe(p)
        why = ""
        if not info:
            why = "unreadable"
        elif BAD.search(title):
            why = "title: people/graphic/360"
        elif info["w"] < 1280:
            why = f"low resolution {info['w']}x{info['h']}"
        elif info["w"] * 9 != info["h"] * 16:
            why = f"not 16:9 ({info['w']}x{info['h']})"
        elif info["seconds"] < 4:
            why = f"too short {info['seconds']}s"
        if why:
            bad.append({"file": p.name, "why": why})
            continue
        words = [w for w in re.findall(r"[a-z]+", title.lower()) if w not in STOP and len(w) > 2]
        ok.append({"file": p.name, "title": title, **info, "keywords": list(dict.fromkeys(words))[:12]})
    OUT.write_text(yaml.safe_dump({"count": len(ok), "source": str(SRC), "clips": ok, "rejected": bad},
                                  sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"usable {len(ok)}  rejected {len(bad)} -> {OUT}")
    from collections import Counter
    print(Counter(b["why"].split(" ")[0] + " " + b["why"].split(" ")[1] if " " in b["why"] else b["why"] for b in bad).most_common(8))


if __name__ == "__main__":
    main()
