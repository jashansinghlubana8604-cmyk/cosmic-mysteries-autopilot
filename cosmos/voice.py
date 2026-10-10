"""Narration with Microsoft's neural voices (edge-tts, free), word-timed.

Each piece of narration is synthesized separately so the edit controls every pause exactly. Pronunciation fixes
from the script's `say:` line are applied to the spoken text only; word timings are mapped back onto the
written words, so on-screen text keeps the real spelling.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .media import FFMPEG

log = logging.getLogger("cosmos.voice")
SR = 48000
if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())   # quiet proactor shutdown noise


@dataclass
class Spoken:
    wav: Path
    duration: float          # seconds of audio (trailing silence trimmed)
    words: list[dict]        # [{"text", "start", "end"}] aligned to the WRITTEN words


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def _apply_say(text: str, say: dict[str, str]) -> str:
    for k, v in say.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


async def _edge(text: str, voice: str, rate: str, pitch: str, mp3: Path) -> list[dict]:
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    words = []
    with open(mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                s = chunk["offset"] / 1e7
                words.append({"text": chunk["text"], "start": s, "end": s + chunk["duration"] / 1e7})
    return words


def _align(written: list[str], spoken_words: list[dict], say: dict[str, str]) -> list[dict]:
    """Give every written word a time, matching the TTS word boundaries in order."""
    say_norm = {_norm(k): [_norm(x) for x in v.split()] for k, v in say.items()}
    targets = []
    for w in written:
        n = _norm(w)
        targets.append(say_norm.get(n, [n])[0] if n else "")
    sp = [_norm(x["text"]) for x in spoken_words]
    out: list[dict | None] = [None] * len(written)
    j = 0
    for i, t in enumerate(targets):
        if not t:
            continue
        for k in range(j, min(j + 6, len(sp))):
            if sp[k] == t or (len(t) > 3 and (sp[k].startswith(t[:4]) or t.startswith(sp[k][:4]) and sp[k])):
                out[i] = {"text": written[i], "start": spoken_words[k]["start"], "end": spoken_words[k]["end"]}
                j = k + 1
                break
    # interpolate the ones the voice merged or split (numbers, initials...)
    known = [(i, o) for i, o in enumerate(out) if o]
    if not known:
        total = spoken_words[-1]["end"] if spoken_words else 1.0
        return [{"text": w, "start": total * i / len(written), "end": total * (i + 1) / len(written)}
                for i, w in enumerate(written)]
    res = []
    for i, w in enumerate(written):
        if out[i]:
            res.append(out[i])
            continue
        prev = max((k for k, _ in known if k < i), default=None)
        nxt = min((k for k, _ in known if k > i), default=None)
        a = out[prev]["end"] if prev is not None else 0.0
        b = out[nxt]["start"] if nxt is not None else (spoken_words[-1]["end"] if spoken_words else a + 0.3)
        lo = prev if prev is not None else -1
        hi = nxt if nxt is not None else len(written)
        f0 = (i - lo) / (hi - lo)
        f1 = (i - lo + 0.9) / (hi - lo)
        res.append({"text": w, "start": a + (b - a) * f0 - 0.0, "end": a + (b - a) * min(1, f1)})
    return res


def synthesize(text: str, out: Path, cfg: dict, say: dict[str, str] | None = None) -> Spoken:
    import json
    say = say or {}
    out.parent.mkdir(parents=True, exist_ok=True)
    mp3 = out.with_suffix(".mp3")
    spoken_text = _apply_say(text, say)
    meta = out.with_suffix(".json")
    sig = {"text": text, "say": spoken_text, "voice": cfg["edge_voice"], "rate": cfg.get("rate"),
           "pitch": cfg.get("pitch")}
    if meta.exists() and out.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("sig") == sig:
                return Spoken(out, m["duration"], m["words"])
        except (ValueError, KeyError):
            pass
    last = None
    for attempt in range(4):
        try:
            words = asyncio.run(_edge(spoken_text, cfg["edge_voice"], cfg.get("rate", "+0%"),
                                      cfg.get("pitch", "+0Hz"), mp3))
            if words and mp3.stat().st_size > 2000:
                break
            raise RuntimeError("no audio returned")
        except Exception as e:  # noqa: BLE001 — network hiccups
            last = e
            log.warning("voice attempt %d failed: %s", attempt + 1, e)
            import time
            time.sleep(2 + 3 * attempt)
    else:
        raise RuntimeError(f"edge-tts failed: {last}")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(mp3), "-ar", str(SR), "-ac", "1", str(out)], check=True)
    written = text.split()
    aligned = _align(written, words, say)
    duration = max(words[-1]["end"] + 0.12, aligned[-1]["end"] + 0.05)
    meta.write_text(json.dumps({"sig": sig, "duration": duration, "words": aligned}), encoding="utf-8")
    for stale in (out.with_name(out.stem + "_p.wav"),):
        stale.unlink(missing_ok=True)
    return Spoken(out, duration, aligned)


async def _gather(jobs, cfg, say, limit=4):
    sem = asyncio.Semaphore(limit)
    loop = asyncio.get_running_loop()

    async def one(text, out):
        async with sem:
            return await loop.run_in_executor(None, synthesize, text, out, cfg, say)
    return await asyncio.gather(*(one(t, o) for t, o in jobs))


def synthesize_many(jobs: list[tuple[str, Path]], cfg: dict, say: dict[str, str]) -> list[Spoken]:
    """Synthesize several pieces in parallel (edge-tts is network bound)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(lambda j: synthesize(j[0], j[1], cfg, say), jobs))
