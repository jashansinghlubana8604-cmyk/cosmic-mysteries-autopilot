"""YouTube title, description (hook, chapters, credits, music licence), tags."""
from __future__ import annotations

import re

from .script import Episode
from . import sound


def _ts(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _hook(ep: Episode) -> str:
    cold = next(s for s in ep.sections if s.kind == "cold")
    text = " ".join(p.text for p in cold.paragraphs)
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sents:
        if len(out) + len(s) > 230:
            break
        out = f"{out} {s}".strip()
    return out


def _credit_summary(credits: list[str]) -> list[str]:
    seen, out = set(), []
    for c in credits:
        c = re.sub(r"\s+", " ", c).strip()
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    return out


def description(ep: Episode, chapters: list[tuple[float, str]], credits: list[str], music: list[str],
                channel: str) -> str:
    parts = [_hook(ep), ""]
    if ep.meta.get("blurb"):
        parts += [ep.meta["blurb"], ""]
    if len(chapters) >= 3:
        parts.append("Chapters")
        parts += [f"{_ts(t)} {name}" for t, name in chapters]
        parts.append("")
    parts.append("A new deep-space mystery every day. Subscribe so you never miss one.")
    parts.append("")
    cr = _credit_summary(credits)
    if cr:
        parts.append("Footage & images (with thanks):")
        line = "; ".join(cr)
        if len(line) > 1700:
            line = line[:1700].rsplit(";", 1)[0] + "; and others."
        parts.append(line)
        parts.append("ESA/Hubble, ESA/Webb, ESO and NOIRLab material is used under CC BY 4.0. "
                     "NASA material is in the public domain.")
        parts.append("")
    tracks = list(dict.fromkeys(m for m in music))
    lines = [sound.credit_line(t) for t in tracks]
    lines = [x for x in lines if x]
    if lines:
        parts.append("Music:")
        parts += lines
        parts.append("")
    tags = ["#space", "#astronomy", "#spacemystery"]
    parts.append(" ".join(tags))
    text = "\n".join(parts)
    return text[:4900]


def tags(ep: Episode) -> list[str]:
    base = ep.tags + ["space mysteries", "space documentary", "universe", "astronomy", "cosmos",
                      "unexplained", "space facts", "deep space"]
    out, total = [], 0
    for t in dict.fromkeys(base):
        if total + len(t) + 1 > 480:
            break
        out.append(t)
        total += len(t) + 1
    return out
