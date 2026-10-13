"""The editor: turns a parsed episode into a finished video.

1. Narration: every piece synthesized, laid out with designed pauses.
2. Visuals: each paragraph's searches -> best unused footage (videos preferred), downloaded in parallel,
   cut into shots that change on the narrator's breaths.
3. Structure: cold open -> warp-speed title -> chapter cards -> chapters -> outro -> end card.
4. Overlays: kinetic on-screen text exactly when the words are spoken (with a spotlight dim and a hit),
   chapter titles, source credits, brand mark.
5. Sound: drone + riser + booms + whooshes, music by mood, ducking, -14 LUFS master.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import subprocess

import cv2
import numpy as np

from . import sound, sources, vision
from .media import FFMPEG, iter_frames, probe
from .render import FPS, H, W, RenderError, Shot, render
from .script import Episode, Paragraph, Section
from .voice import Spoken, synthesize

log = logging.getLogger("cosmos.edit")

COLD_LEAD = 1.0
GAP_SENT = 0.45        # breath between sentences
GAP_PAUSE = 1.5        # [pause] in the script: let the line land
GAP_PARA = 1.1         # between paragraphs: the picture breathes, the music swells
TITLE_WARP = 3.2       # drift through the stars
TITLE_HOLD = 3.6
CARD_LEN = 3.4
CARD_VOICE_AT = 2.7
END_CARD = 12.0
# majestic fallbacks: used when a paragraph's own searches are thin, and mixed in where they are more beautiful
GENERIC = ["v: zoom nebula", "v: pan nebula", "v: flight through galaxy", "v: zoom galaxy", "i: nebula",
           "i: spiral galaxy", "i: star forming region", "v: milky way time-lapse", "i: pillars", "v: fly through",
           # moving footage first: a deep pool of real space video every paragraph can fall back on
           "v: zoom", "v: fly-through", "v: 3d visualisation", "v: pan", "v: artist's animation",
           "v: deep space", "v: star cluster", "v: galaxy cluster", "v: webb zoom", "v: hubble zoom",
           "v: supernova animation", "v: black hole animation", "v: exoplanet animation", "v: planet animation"]
EARTHLY = re.compile(r"observator|\bdome|\bsite\b|telescope|antenna|instrument|mirror|camera|laborator|"
                     r"platform|building|\broad\b|rocket|launch|satellite|spacecraft|construction|control room|"
                     r"aerial|paranal|la silla|armazones|\balma\b|\bvlt\b|\belt\b|\bcta\b|\bvista\b|hotel|"
                     r"residencia|engineer|workshop|assembly|clean room|\btest|prototype|\bmodel\b", re.I)
AWE = re.compile(r"nebula|galax|cluster|pillars|supernova|remnant|zoom|\bpan\b|panning|fly|flight|journey|"
                 r"milky way|deep field|black hole|quasar|star-forming|star forming|stellar nursery|aurora|"
                 r"cosmic|universe|interstellar|exoplanet|planet|comet|jet|disc|disk|eclipse|webb|hubble|"
                 r"artist's impression|artist’s impression|simulation|visuali[sz]ation", re.I)
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


# ---- narration -----------------------------------------------------------------------------------------
@dataclass
class Piece:
    sec: int
    para: int
    part: int
    text: str
    emphasis: list
    spoken: Spoken | None = None
    t: float = 0.0
    sent: int = 0
    last_in_part: bool = True


def _sentences(text: str) -> list[tuple[str, int]]:
    """Split into sentences, returning (sentence, index of its first word in the text)."""
    words = text.split()
    out, cur, start = [], [], 0
    for i, w in enumerate(words):
        cur.append(w)
        end = re.search(r"[.!?][\"')\]]*$", w) and not re.match(r"^(Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e)\.$", w)
        if end or i == len(words) - 1:
            out.append((" ".join(cur), start))
            cur, start = [], i + 1
    # glue very short fragments ("Nothing.") to the next sentence only if they are a single word without stop
    return out


def _voice_all(ep: Episode, work: Path, vcfg: dict, next_teaser: str) -> list[Piece]:
    pieces = []
    for si, s in enumerate(ep.sections):
        for pi, para in enumerate(s.paragraphs):
            for ki, part in enumerate(para.parts):
                text = part.text.replace("{next}", next_teaser)
                sents = _sentences(text)
                for k, (st, w0) in enumerate(sents):
                    n = len(st.split())
                    emph = []
                    for e in part.emphasis:
                        if w0 <= e.first_word < w0 + n:
                            emph.append(type(e)(e.spoken, e.shown, e.first_word - w0, e.n_words))
                    pieces.append(Piece(si, pi, ki, st, emph, sent=k, last_in_part=k == len(sents) - 1))
    jobs = [(p.text, work / "voice" / f"s{p.sec:02d}_p{p.para:02d}_{p.part}_{p.sent:02d}.wav") for p in pieces]
    say = ep.pronunciations

    def one(j):
        return synthesize(j[0], j[1], vcfg, say)
    with ThreadPoolExecutor(2) as ex:      # the new TTS host answers 429 to bigger bursts
        res = list(ex.map(one, jobs))
    for p, r in zip(pieces, res):
        p.spoken = r
    return pieces


# ---- footage ---------------------------------------------------------------------------------------------
@dataclass
class Media:
    asset: sources.Asset
    path: Path
    kind: str
    duration: float = 0.0
    beauty: float = 0.5
    bright: float = 60.0
    scenes: list = field(default_factory=list)
    uses: list = field(default_factory=list)          # [(timeline_t, src_a, src_b)]


class Footage:
    def __init__(self, cache: Path, seed: int) -> None:
        self.cache = cache
        self.media: dict[str, Media] = {}
        self.bad: set[str] = set()
        self.rng = random.Random(seed)
        self.search_cache: dict[str, list[sources.Asset]] = {}
        self._beauty: dict[str, float] = {}
        self._bright: dict[str, float] = {}

    def search(self, q: str) -> list[sources.Asset]:
        if q not in self.search_cache:
            try:
                self.search_cache[q] = sources.search(q, self.cache)
            except Exception as e:  # noqa: BLE001
                log.warning("search failed for %r: %s", q, e)
                self.search_cache[q] = []
        return self.search_cache[q]

    def search_all(self, queries: list[str]) -> None:
        sources._Index.load()                # load the archive index once, before the worker threads
        todo = list(dict.fromkeys(q for q in queries if q not in self.search_cache))
        with ThreadPoolExecutor(8) as ex:
            for q, res in zip(todo, ex.map(self._safe_search, todo)):
                self.search_cache[q] = res

    def _safe_search(self, q: str) -> list[sources.Asset]:
        try:
            return sources.search(q, self.cache)
        except Exception as e:  # noqa: BLE001
            log.warning("search failed for %r: %s", q, e)
            return []

    def candidates(self, queries: list[str], per: int = 10) -> list[sources.Asset]:
        lists = [self.search(q)[:per] for q in queries]
        out, seen = [], set()
        for i in range(per):
            for lst in lists:
                if i < len(lst) and lst[i].key not in seen:
                    seen.add(lst[i].key)
                    out.append(lst[i])
        return out

    def _looks_ok(self, a: sources.Asset, frames: list[np.ndarray]) -> bool:
        """No people, no burned-in text (cached per asset)."""
        cf = self.cache / "vision" / (re.sub(r"[^\w\-]", "_", a.key) + ".json")
        if cf.exists():
            d = json.loads(cf.read_text())
            if "beauty" in d and "bright" in d and d.get("v") == 2:
                self._beauty[a.key] = d["beauty"]
                self._bright[a.key] = d["bright"]
                return d["ok"]
        ok, why = True, "ok"
        if all(vision.is_dark(f) for f in frames):
            ok, why = False, "too dark"
        for f in frames if ok else []:
            good, why = vision.verdict(f)
            if not good:
                ok = False
                break
        beauty = float(np.mean([vision.beauty(f) for f in frames])) if frames else 0.5
        bright = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames])) if frames else 60.0
        if ok and bright < 11:
            ok, why = False, f"too dark ({bright:.0f})"
        self._beauty[a.key] = beauty
        self._bright[a.key] = bright
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps({"ok": ok, "why": why, "beauty": beauty, "bright": bright, "v": 2}))
        if not ok:
            log.info("rejected %s (%s): %s", a.key, why, a.title[:60])
        return ok

    def _validate(self, a: sources.Asset, path: Path) -> Media | None:
        try:
            if a.kind == "video":
                info = probe(path)
                if info["duration"] < 2.5 or info["width"] < 900:
                    return None
                m = Media(a, path, "video", info["duration"])
                m.scenes = self._scenes(a, path, info)
                if not m.scenes:
                    return None
                picks = [m.scenes[0], m.scenes[len(m.scenes) // 2], m.scenes[-1]]
                frames = []
                for s0, s1 in dict.fromkeys(picks):
                    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{(s0 + s1) / 2:.2f}", "-i", str(path),
                                          "-frames:v", "1", "-vf", "scale=960:-2", "-f", "image2pipe", "-vcodec", "png",
                                          "-"], capture_output=True).stdout
                    f = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
                    if f is not None:
                        frames.append(f)
                return m if frames and self._looks_ok(a, frames) else None
            img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
            if img is None:
                return None
            h, w = img.shape[:2]
            if min(w * 2, h * 2) < 700:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (gray > 235).mean() > 0.35:       # diagrams / charts on white
                return None
            if gray.std() < 6:
                return None
            if not self._looks_ok(a, [img]):
                return None
            return Media(a, path, "image")
        except Exception as e:  # noqa: BLE001
            log.info("invalid media %s: %s", a.key, e)
            return None

    def _frame_at(self, path: Path, t: float) -> np.ndarray | None:
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                              "-vf", "scale=960:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None

    def _scan(self, path: Path, info: dict, lo: float, hi: float, every: float = 2.5) -> list[tuple[float, np.ndarray]]:
        """(time, frame) about every `every` seconds between lo and hi, in one fast pass that decodes keyframes only."""
        w = 960
        h = max(2, int(round(w * info["height"] / max(1, info["width"]) / 2)) * 2)
        cmd = [FFMPEG, "-v", "error", "-skip_frame", "nokey", "-ss", f"{lo:.2f}", "-i", str(path),
               "-t", f"{max(0.1, hi - lo):.2f}", "-an", "-vf", f"fps={1 / every:.4f},scale={w}:{h}",
               "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        size = w * h * 3
        return [(lo + k * every, np.frombuffer(raw[k * size:(k + 1) * size], np.uint8).reshape(h, w, 3))
                for k in range(len(raw) // size)]

    def _clean_scenes(self, path: Path, scenes: list[tuple[float, float]], info: dict) -> list[tuple[float, float]]:
        """Drop every scene that shows writing anywhere in it (checked about every 2.5 s): subtitles, captions,
        labels that fade in mid-scene, logo end cards with web addresses. A mostly captioned clip is dropped whole."""
        if not scenes:
            return []
        frames = self._scan(path, info, scenes[0][0], scenes[-1][1])
        if not frames:          # keyframe pass failed: fall back to three seeks per scene
            frames = [(t, f) for s0, s1 in scenes for t in (s0 + 0.3, (s0 + s1) / 2, s1 - 0.3)
                      for f in [self._frame_at(path, t)] if f is not None]
        bad = [t for t, f in frames if vision.has_text(f)]
        keep = [(s0, s1) for s0, s1 in scenes if not any(s0 - 1.0 <= t <= s1 + 1.0 for t in bad)]
        return keep if len(keep) * 2 >= len(scenes) else []

    def _raw_scenes(self, a: sources.Asset, path: Path, info: dict) -> list[tuple[float, float]]:
        """Shots between hard cuts, skipping the first and last few percent and black sections (cached)."""
        cf = self.cache / "scenes" / (re.sub(r"[^\w\-]", "_", a.key) + ".json")
        if cf.exists():
            return [tuple(x) for x in json.loads(cf.read_text())]
        dur = info["duration"]
        lo = max(1.2, dur * 0.07)
        hi = dur - max(2.5, dur * 0.12)
        if dur < 8:
            lo, hi = 0.4, dur - 0.6
        diffs, prev, means = [], None, []
        for f in iter_frames(path, 64, 36, gray=True, duration=min(dur, 240), vf_prefix="fps=12.5,"):
            f = f.astype(np.int16)
            diffs.append(0.0 if prev is None else float(np.mean(np.abs(f - prev))))
            means.append(float(f.mean()))
            prev = f
        rate = 12.5
        cuts = [lo]
        if len(diffs) > 3:
            d = np.array(diffs)
            thr = max(14.0, 5 * float(np.median(d[1:]) + 0.5))
            cuts += [i / rate for i in range(1, len(d) - 1) if d[i] > thr and d[i] >= d[i - 1] and d[i] >= d[i + 1]
                     and lo < i / rate < hi]
        cuts.append(hi)
        scenes = []
        for x, y in zip(cuts, cuts[1:]):
            x, y = x + 0.15, y - 0.15
            if y - x < 1.6:
                continue
            seg = means[int(x * rate):int(y * rate)]
            if seg and float(np.median(seg)) < 4:        # black / fade sections
                continue
            scenes.append((round(x, 2), round(y, 2)))
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(scenes))
        return scenes

    def _scenes(self, a: sources.Asset, path: Path, info: dict) -> list[tuple[float, float]]:
        """Usable shots: raw shots, minus the labelled opening of zoom videos, minus any shot with writing in it."""
        cf = self.cache / "scenes" / (re.sub(r"[^\w\-]", "_", a.key) + ".v5.json")
        if cf.exists():
            return [tuple(x) for x in json.loads(cf.read_text())]
        scenes = self._raw_scenes(a, path, info)
        dur = info["duration"]
        if "zoom" in (a.title or "").lower() and dur >= 20:
            lo = dur * 0.45             # zoom videos open on labelled constellation maps: keep only the close-up
            scenes = [(max(x, lo), y) for x, y in scenes if y - max(x, lo) >= 1.6]
        scenes = self._clean_scenes(path, scenes, info)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(scenes))
        return scenes

    def fetch(self, assets: list[sources.Asset]) -> None:
        todo = [a for a in assets if a.key not in self.media and a.key not in self.bad]
        if not todo:
            return

        def one(a):
            p = sources.download(a, self.cache)
            return a, p
        with ThreadPoolExecutor(6) as ex:
            for a, p in ex.map(one, todo):
                if p is None:
                    self.bad.add(a.key)
                    continue
                m = self._validate(a, p)
                if m is None:
                    self.bad.add(a.key)
                else:
                    m.beauty = self._beauty.get(a.key, 0.5)
                    m.bright = self._bright.get(a.key, 60.0)
                    self.media[a.key] = m

    def window(self, m: Media, length: float, t: float) -> tuple[float, float] | None:
        """A (start, speed) inside one scene, not overlapping earlier uses of this clip."""
        opts = []
        slow = 1.15                                   # gentle slow motion: space should feel vast and calm
        for a, b in m.scenes:
            span = b - a
            speed = slow
            if span * slow < length:
                speed = length / span
                if speed > 1.8:
                    continue
            need = length / speed
            for k in range(6):
                s = a + (span - need) * (k / 5 if span > need else 0)
                e = s + need
                clash = any(not (e <= u0 or s >= u1) for _, u0, u1 in m.uses)
                if not clash:
                    opts.append((abs(speed - slow), k, s, speed, e))
        if not opts:
            return None
        opts.sort(key=lambda o: (o[0], self.rng.random()))
        _, _, s, speed, e = opts[0]
        m.uses.append((t, s, e))
        return s, speed


# ---- the episode ---------------------------------------------------------------------------------------------
@dataclass
class Result:
    video: Path
    duration: float
    chapters: list[tuple[float, str]]
    credits: list[str]
    music: list[str]
    hero: Path | None
    srt: Path
    stills: list[Path]
    thumbs: list[Path] = field(default_factory=list)     # images matching the thumbnail searches


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines)


ASS_HEAD = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Big,Bebas Neue,170,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,6,0,1,0,0,5,80,80,0,1
Style: Glow,Bebas Neue,170,&H00FFD38A,&H00FFD38A,&H00FFD38A,&H00000000,0,0,0,0,100,100,6,0,1,6,0,5,80,80,0,1
Style: Kicker,Montserrat ExtraBold,34,&H00FFD38A,&H00FFD38A,&H00000000,&H00000000,0,0,0,0,100,100,14,0,1,0,0,5,0,0,0,1
Style: Title,Bebas Neue,150,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,8,0,1,0,0,5,120,120,0,1
Style: TitleGlow,Bebas Neue,150,&H00FFC060,&H00FFC060,&H00FFC060,&H00000000,0,0,0,0,100,100,8,0,1,8,0,5,120,120,0,1
Style: Credit,Montserrat ExtraBold,19,&H50FFFFFF,&H50FFFFFF,&H90000000,&H00000000,0,0,0,0,100,100,1,0,1,1,0,1,44,40,30,1
Style: Mark,Bebas Neue,34,&H70FFFFFF,&H70FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,6,0,1,0,0,9,0,60,40,1
Style: Line,Bebas Neue,10,&H00FFD38A,&H00FFD38A,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class Overlay:
    def __init__(self) -> None:
        self.ev: list[str] = []

    def add(self, a: float, b: float, style: str, text: str, layer: int = 1) -> None:
        if b - a > 0.05:
            self.ev.append(f"Dialogue: {layer},{_ass_time(a)},{_ass_time(b)},{style},,0,0,0,,{text}")

    def big(self, a: float, b: float, text: str, y: int = 540, size: int = 170) -> None:
        t = _wrap(_esc(text.upper()), 18)
        dur = int((b - a) * 1000)
        # slow reveal: fades in out of a soft blur, letters drift gently apart (no punch)
        anim = rf"\an5\pos({W // 2},{y})\fs{size}\fad(700,900)"
        self.add(a, b, "Glow", rf"{{{anim}\blur26\alpha&H60&\fsp10\t(0,{dur},\fsp22)}}{t}", 5)
        self.add(a, b, "Big", rf"{{{anim}\fsp10\t(0,{dur},\fsp22)\blur6\t(0,900,\blur0.6)}}{t}", 6)

    def card(self, a: float, b: float, kicker: str, title: str, y: int = 560) -> None:
        t = _wrap(_esc(title.upper()), 22)
        lines = t.count(r"\N") + 1
        if kicker:
            self.add(a, b, "Kicker", rf"{{\an5\pos({W // 2},{y - 70 - (lines - 1) * 70})\fad(900,700)\fsp20\t(0,{int((b - a) * 1000)},\fsp30)}}"
                     f"{_esc(kicker.upper())}", 6)
        dur = int((b - a) * 1000)
        anim = rf"\an5\pos({W // 2},{y + 40 - (lines - 1) * 10})\fad(1100,800)\fsp6\t(0,{dur},\fsp16)"
        self.add(a + 0.3, b, "TitleGlow", rf"{{{anim}\blur30\alpha&H68&}}{t}", 5)
        self.add(a + 0.3, b, "Title", rf"{{{anim}\blur10\t(0,1400,\blur0.5)}}{t}", 6)
        ly = y + 40 + lines * 72 + 10 - (lines - 1) * 10
        self.add(a + 0.6, b, "Line", rf"{{\an5\pos({W // 2},{ly})\fad(0,400)\fscx0\t(0,800,0.5,\fscx100)\p1}}"
                 r"m -180 0 l 180 0 l 180 3 l -180 3{\p0}", 6)


def produce(ep: Episode, cfg: dict, work: Path, cache: Path, *, seed: int, channel: str, next_teaser: str,
            log_fn=print) -> Result:
    rng = random.Random(seed)
    work.mkdir(parents=True, exist_ok=True)
    vcfg = cfg["voice"]
    t0 = time.time()

    # 1) narration ------------------------------------------------------------------------------------------
    pieces = _voice_all(ep, work, vcfg, next_teaser)
    log_fn(f"narration: {len(pieces)} pieces, {sum(p.spoken.duration for p in pieces) / 60:.1f} min "
           f"({time.time() - t0:.0f}s)")

    # 2) timeline in seconds ------------------------------------------------------------------------------
    by_para: dict[tuple[int, int], list[Piece]] = {}
    for p in pieces:
        by_para.setdefault((p.sec, p.para), []).append(p)

    def layout(k: float):
        """The timeline with every breath/pause/paragraph gap scaled by k."""
        blocks = []        # (kind, t0, t1, data)
        t = 0.0
        chapters: list[tuple[float, str]] = [(0.0, "Intro")]
        music_plan: list[list] = []
        title_t = None
        chap_no = 0
        cold_end = 0.0
        for si, sec in enumerate(ep.sections):
            if sec.kind == "title":
                t += 0.4
                title_t = t
                blocks.append(("title", t, t + TITLE_WARP + TITLE_HOLD, None))
                t += TITLE_WARP + TITLE_HOLD
                continue
            if sec.kind == "chapter":
                chap_no += 1
                blocks.append(("card", t, t + CARD_LEN, (si, chap_no)))
                chapters.append((t, sec.title))
                voice_t = t + CARD_VOICE_AT
            elif sec.kind == "cold":
                voice_t = t + COLD_LEAD
            else:  # outro
                voice_t = t + 0.6
            sec_start = t
            t = voice_t
            for pi, para in enumerate(sec.paragraphs):
                ps = by_para[(si, pi)]
                p_start = t
                for j, pc in enumerate(ps):
                    pc.t = t
                    t += pc.spoken.duration
                    if j < len(ps) - 1:
                        t += (GAP_SENT if not pc.last_in_part else GAP_PAUSE) * k
                t += GAP_PARA * k
                blocks.append(("para", p_start, t, (si, pi)))
            if sec.kind == "cold":
                t += 0.5
                cold_end = t
            elif sec.kind == "outro":
                t += 0.4
            music_plan.append([sec_start, t, sec.mood or ep.mood, sec.kind])
        end_t = t
        blocks.append(("end", t, t + END_CARD, None))
        total = t + END_CARD
        return blocks, chapters, music_plan, title_t, cold_end, end_t, total

    sfx: list[tuple[float, np.ndarray, float, float]] = []
    ov = Overlay()
    min_total = float(cfg.get("render", {}).get("min_minutes", 8.15)) * 60
    blocks, chapters, music_plan, title_t, cold_end, end_t, total = layout(1.0)
    if total < min_total:      # never publish under 8:00 (mid-roll ads): let the picture breathe longer
        fixed = layout(0.0)[-1]
        k = min(2.4, (min_total - fixed) / max(1e-6, total - fixed))
        log_fn(f"script runs {total / 60:.2f} min; stretching pauses x{k:.2f} to reach {min_total / 60:.2f} min")
        blocks, chapters, music_plan, title_t, cold_end, end_t, total = layout(k)

    # visual coverage: each paragraph's slot runs from slightly before its narration to the next block
    slots = []          # [t0, t1, (si, pi) | special]
    for k, (kind, a, b, data) in enumerate(blocks):
        nxt = blocks[k + 1][1] if k + 1 < len(blocks) else total
        if kind == "para":
            si, pi = data
            prev_kind = blocks[k - 1][0] if k else None
            a0 = a - (0.35 if prev_kind == "para" else 0.0)
            if prev_kind in ("card",):
                a0 = blocks[k - 1][2]
            if prev_kind is None:
                a0 = 0.0
            slots.append([a0, nxt - (0.35 if k + 1 < len(blocks) and blocks[k + 1][0] == "para" else 0.0), data])
        else:
            slots.append([a, b if kind != "end" else total, kind])
    # make slots contiguous
    for k in range(len(slots) - 1):
        slots[k][1] = slots[k + 1][0]
    slots[-1][1] = total
    if slots[0][0] > 0:
        slots[0][0] = 0.0

    # 3) footage ----------------------------------------------------------------------------------------------
    fx = Footage(cache, seed)
    fallback = ep.fallback
    fx.search_all([q for s in ep.sections for p in s.paragraphs for q in p.queries]
                  + fallback + ep.thumb_queries + GENERIC)
    awe = fx.candidates(GENERIC, per=12)            # the beautiful pool every paragraph may draw on
    awe.sort(key=lambda c: c.kind != "video")       # videos first, so they are the ones downloaded
    plans = []
    for a, b, data in slots:
        if not isinstance(data, tuple):
            plans.append((a, b, data, 0, [], []))
            continue
        si, pi = data
        sec = ep.sections[si]
        para = sec.paragraphs[pi]
        target = 4.8 if sec.kind == "cold" else 6.2   # cut pace of the winners
        n = max(1, round((b - a) / target))
        plans.append((a, b, data, n, fx.candidates(para.queries), para.queries))
    want: list[sources.Asset] = []
    for a, b, data, n, cands, _q in plans:
        want += cands[: n + 3]
    want += fx.candidates(fallback + ep.thumb_queries, per=6) + awe[:90]
    seen = set()
    want = [x for x in want if not (x.key in seen or seen.add(x.key))]
    log_fn(f"footage: downloading {len(want)} candidates")
    fx.fetch(want)
    log_fn(f"footage: {len(fx.media)} usable ({sum(1 for m in fx.media.values() if m.kind == 'video')} videos) "
           f"({time.time() - t0:.0f}s)")

    used_at: dict[str, float] = {}
    earthly_in_para: dict[tuple, int] = {}

    def rank(cands: list[sources.Asset], queries: list[str], extra: list[sources.Asset] = (),
             early: bool = False) -> list:
        """(total, asset) best first: relevance + beauty; ground/hardware shots only when asked for.
        `early` (first minutes): the most stunning, moving footage wins, so viewers are hooked at once."""
        qtext = " ".join(queries).lower()
        wants_ground = bool(EARTHLY.search(qtext))
        out, seen_k = [], set()
        for base, lst in ((0.0, cands), (-0.05 if early else -0.32, extra)):
            for c in lst:
                m = fx.media.get(c.key)
                if not m or c.key in seen_k:
                    continue
                seen_k.add(c.key)
                s = c.score + base + (0.8 if early else 0.3) * m.beauty
                if AWE.search(c.title):
                    s += 0.3
                if m.bright < 22:
                    s -= 0.6 if m.bright < 16 else 0.3
                if m.kind == "video":            # real motion beats a still everywhere; in the opening by far
                    s += 1.1 if early else 0.8
                if EARTHLY.search(c.title) and not wants_ground:
                    s -= 0.5
                if m.kind == "video" and re.search(r"zoom|pan|fly|flight|journey|travel|tour|3d|rotat", c.title, re.I):
                    s += 0.12
                out.append((s, c))
        out.sort(key=lambda x: -x[0])
        return out

    def choose(ranked: list, at: float, length: float, para_key=None):
        """Best media for a shot at time `at`: fresh first; reuse only after a long gap."""
        for fresh in (True, False):
            for _, c in ranked:
                m = fx.media.get(c.key)
                last = used_at.get(c.key)
                if fresh and last is not None:
                    continue
                if not fresh and last is not None and at - last < (35 if m.kind == "video" else 150):
                    continue
                ground = bool(EARTHLY.search(c.title))
                if para_key is not None and ground and earthly_in_para.get(para_key, 0) >= 1:
                    continue
                if m.kind == "video":
                    w = fx.window(m, length, at)
                    if w is None:
                        continue
                    res = (m, w)
                else:
                    res = (m, None)
                used_at[c.key] = at
                if para_key is not None and ground:
                    earthly_in_para[para_key] = earthly_in_para.get(para_key, 0) + 1
                return res
        return None

    def ensure(cands: list[sources.Asset], k: int) -> None:
        """Download more candidates if fewer than k fresh usable ones are ready."""
        ready = sum(1 for c in cands if c.key in fx.media and c.key not in used_at)
        if ready >= k:
            return
        more = [c for c in cands if c.key not in fx.media and c.key not in fx.bad][: (k - ready) + 3]
        fx.fetch(more)

    shots: list[Shot] = []
    shot_t: list[float] = []
    credits_used: list[str] = []
    stills: list[Path] = []
    moves_img = ["push", "pan_r", "pull", "float", "pan_l", "tilt_u", "push", "tilt_d"]
    mi = rng.randrange(len(moves_img))
    hero_cands = fx.candidates([q if q.startswith(("i:", "c:")) else f"i: {q.split(':', 1)[-1].strip()}"
                                for q in ep.thumb_queries], per=10)
    fx.fetch(hero_cands[:14])
    from . import thumbnail
    thumbs = thumbnail.candidates(ep, fx)
    hero: Path | None = thumbnail.choose(ep, thumbs) or (thumbs[0] if thumbs else None)
    if hero is None:
        imgs = sorted((m for m in fx.media.values() if m.kind == "image"), key=lambda m: -m.beauty)
        hero = imgs[0].path if imgs else None
    if hero is not None:
        for k_, m_ in fx.media.items():
            if m_.path == hero:
                used_at[k_] = -1000.0

    def frames_of(sec_len: float) -> int:
        return max(1, int(round(sec_len * FPS)))

    word_gaps = sorted(p.t + p.spoken.duration + 0.15 for p in pieces)   # sentence ends: natural cut points

    def snap(x: float, lo: float, hi: float) -> float:
        best = None
        for g in word_gaps:
            if lo + 3.0 <= g <= hi - 3.0 and abs(g - x) <= 2.0:
                if best is None or abs(g - x) < abs(best - x):
                    best = g
        return best if best is not None else x

    def add_shot(sh: Shot, at: float, credit: str) -> None:
        shots.append(sh)
        shot_t.append(at)
        if credit and credit not in credits_used:
            credits_used.append(credit)

    timeline_frames = 0

    def frames_until(t_end: float) -> int:
        return frames_of(t_end) - timeline_frames

    for (a, b, data, n, cands, queries) in plans:
        if data == "title":
            nf_w = frames_until(a + TITLE_WARP)
            add_shot(Shot("warp", nf_w, trans_in="fade", trans_len=36, seed=seed, move="drift", label="warp"), a, "")
            timeline_frames += nf_w
            nf = frames_until(b)
            if hero is not None:
                add_shot(Shot("image", nf, hero, move="push", strength=0.55, trans_in="fade", trans_len=40,
                              dim=0.3, seed=seed + 1, label="title"), a + TITLE_WARP, "")
            else:
                add_shot(Shot("warp", nf, trans_in="fade", trans_len=40, move="drift", seed=seed + 1), a, "")
            timeline_frames += nf
            continue
        if data == "card":
            continue  # handled with the next paragraph (needs its footage)
        if data == "end":
            nf = frames_until(b)
            pool = [(0.3 * m.beauty + (0.4 if AWE.search(m.asset.title) else 0)
                     - (0.8 if EARTHLY.search(m.asset.title) else 0), m.path)
                    for k_, m in fx.media.items() if m.kind == "image" and k_ not in used_at]
            src = max(pool)[1] if pool else hero
            add_shot(Shot("image", nf, src, move="float", strength=0.5, trans_in="fade", trans_len=45,
                          fade_out=75, seed=seed + 99, label="end"), a, "")
            timeline_frames += nf
            continue
        si, pi = data
        sec = ep.sections[si]
        is_cold = sec.kind == "cold"
        card_block = None
        if pi == 0 and sec.kind == "chapter":
            card_block = next(bl for bl in blocks if bl[0] == "card" and bl[3][0] == si)
        ensure(cands, n + (1 if card_block else 0))
        early = a < 150
        ranked = rank(cands, queries, awe + fx.candidates(fallback, per=6), early=early)
        if card_block:
            ca, cb = card_block[1], card_block[2]
            got = choose(ranked, ca, cb - ca + 1)
            nf = frames_until(cb)
            if got:
                m, w = got
                sh = Shot(m.kind, nf, m.path, start=w[0] if w else 0, speed=w[1] if w else 1, move="push",
                          strength=0.45, trans_in="fade", trans_len=40, dim=0.42, soft=0.5, seed=seed + si,
                          label="card")
            else:
                sh = Shot("warp", nf, trans_in="fade", trans_len=40, move="drift", seed=seed + si, label="card")
            add_shot(sh, ca, "")
            timeline_frames += nf
            ov.card(ca + 0.5, cb - 0.2, f"Part {ROMAN[card_block[3][1] - 1]}", sec.title)
            sfx.append((ca - 1.6, sound.swell(seed + si, 2.2), -15.0, 0.0))
            sfx.append((ca + 0.55, sound.deep(seed + si), -13.0, 0.0))
        length = b - a
        bounds = [a + length * k / n for k in range(n + 1)]
        for k in range(1, n):
            bounds[k] = snap(bounds[k], bounds[k - 1], bounds[k + 1] if k + 1 <= n else b)
        for k in range(n):
            s0, s1 = bounds[k], bounds[k + 1]
            got = choose(ranked, s0, s1 - s0, para_key=data)
            if got is None:
                ensure(cands + awe, 4)
                got = choose(rank(cands, queries, awe + fx.candidates(fallback, per=8), early=early), s0, s1 - s0)
            nf = frames_until(s1)
            first = k == 0
            if first and card_block:
                trans, tl = "fade", 40
            elif first and shots and shots[-1].label == "title":
                trans, tl = "fade", 40
            elif first:
                trans, tl = "fade", 40
            else:
                trans, tl = "fade", (34 if not is_cold else 26)
            if not shots:
                trans, tl = "dip", 12
            if got is None:
                sh = Shot("warp", nf, trans_in=trans, trans_len=tl, move="drift", seed=seed + len(shots))
                cr = ""
            else:
                m, w = got
                if m.kind == "video":
                    sh = Shot("video", nf, m.path, start=w[0], speed=w[1], move=rng.choice(["push", "pull"]),
                              strength=0.45, trans_in=trans, trans_len=tl, seed=seed + len(shots))
                else:
                    mv = moves_img[mi % len(moves_img)]
                    mi += 1
                    sh = Shot("image", nf, m.path, move=mv, strength=0.6 if not is_cold else 0.75,
                              trans_in=trans, trans_len=tl, seed=seed + len(shots))
                    if len(stills) < 60:
                        stills.append(m.path)
                cr = m.asset.credit
            add_shot(sh, s0, cr)          # credits go in the description only: no text on the picture
            timeline_frames += nf

    # 4) on-screen text, spotlight dims, soft hits -----------------------------------------------------------
    emph_events = []
    for p in pieces:
        for e in p.emphasis:
            ws = p.spoken.words
            if e.first_word >= len(ws):
                continue
            a = p.t + ws[e.first_word]["start"] - 0.1
            last = min(len(ws) - 1, e.first_word + e.n_words - 1)
            b = max(a + 2.8, p.t + ws[last]["end"] + 1.8)
            emph_events.append([a, min(b, a + 4.2), e.shown])
    emph_events.sort()
    for k in range(len(emph_events) - 1):
        emph_events[k][1] = min(emph_events[k][1], emph_events[k + 1][0] - 0.1)
    for a, b, text in emph_events:
        size = 180 if len(text) <= 8 else 150 if len(text) <= 16 else 120
        ov.big(a, b, text, size=size)
        reveal = text.endswith(("!", "?")) or (text.isupper() and len(text.split()) <= 2)
        if reveal:
            sfx.append((a - 0.05, sound.deep(seed + int(a * 10)), -14.0, 0.0))
        elif re.search(r"\d", text):
            sfx.append((a, sound.shimmer(seed + int(a * 10), 3.0), -21.0, 0.0))
    starts = []
    acc = 0
    for sh in shots:
        starts.append(acc)
        acc += sh.frames
    for a, b, _ in emph_events:
        fa, fb = int(a * FPS), int(b * FPS)
        for sh, s0 in zip(shots, starts):
            lo, hi = max(fa, s0), min(fb, s0 + sh.frames)
            if lo < hi:
                sh.dims.append((lo - s0, hi - s0, 0.22))

    # title sequence and cold open sound; no channel branding anywhere
    if title_t is not None:
        ov.card(title_t + TITLE_WARP + 0.3, title_t + TITLE_WARP + TITLE_HOLD - 0.5, "", ep.card, y=540)
        sfx.append((title_t + TITLE_WARP - 3.4, sound.swell(seed, 3.6), -11.0, 0.0))
        sfx.append((title_t + TITLE_WARP + 0.2, sound.deep(seed + 7), -9.0, 0.0))
        sfx.append((title_t + TITLE_WARP + 0.4, sound.shimmer(seed + 11, 4.5), -17.0, 0.2))
    if cold_end:
        sfx.append((0.0, sound.drone(cold_end + 2.0, seed), -17.0, 0.0))
        sfx.append((0.6, sound.deep(seed + 3), -15.0, 0.0))

    ass = work / "overlay.ass.src"
    ass.write_text(ASS_HEAD + "\n".join(ov.ev) + "\n", encoding="utf-8")

    # 5) sound ---------------------------------------------------------------------------------------------
    voice_events = [(p.t, p.spoken.wav) for p in pieces]
    plan = []
    for a, b, mood, kind in music_plan:
        if kind == "cold":
            plan.append([a + 0.2, b + 0.3, "dread" if mood in ("dread",) else "mystery"])
        elif plan and plan[-1][2] == mood and kind != "outro":
            plan[-1][1] = b
        else:
            plan.append([a, b, mood])
    # title gap: music restarts with the first chapter; the end card keeps the last section going
    if plan:
        plan[-1][1] = total
    duration = sum(s.frames for s in shots) / FPS
    mixed, music_used = sound.mix(work, duration, voice_events, sfx, [tuple(x) for x in plan], cfg, seed)
    log_fn(f"sound mixed ({time.time() - t0:.0f}s); rendering {duration / 60:.1f} min, {len(shots)} shots")

    # 6) subtitles file (uploaded as captions, not burned in) ---------------------------------------------
    srt = work / "captions.srt"
    srt.write_text(_srt(pieces), encoding="utf-8")

    # 7) picture --------------------------------------------------------------------------------------------
    out = work / "episode.mp4"

    def prog(i, n):
        if i % 3000 == 0:
            log_fn(f"  render {i / n * 100:.0f}% ({time.time() - t0:.0f}s)")
    render(shots, ass, mixed, out, work, seed, crf=int(cfg.get("render", {}).get("crf", 19)),
           preset=cfg.get("render", {}).get("preset", "medium"), progress=prog)
    log_fn(f"rendered in {time.time() - t0:.0f}s")
    (work / "plan.json").write_text(json.dumps([{"t": round(t, 2), "kind": s.kind, "src": str(s.src) if s.src else "",
                                                 "start": round(s.start, 2), "frames": s.frames, "trans": s.trans_in,
                                                 "move": s.move} for s, t in zip(shots, shot_t)], indent=1))
    return Result(out, duration, chapters, credits_used, music_used, hero, srt, stills, thumbs)


def _srt(pieces: list[Piece]) -> str:
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    lines = []
    for p in pieces:
        cur, start = [], None
        ws = p.spoken.words
        for i, w in enumerate(ws):
            if start is None:
                start = p.t + w["start"]
            cur.append(w["text"])
            txt = " ".join(cur)
            end_here = len(txt) > 38 or txt.endswith((".", "?", "!", ",")) and len(txt) > 18 or i == len(ws) - 1
            if end_here:
                lines.append((start, p.t + w["end"] + 0.15, txt))
                cur, start = [], None
    out = []
    for k, (a, b, t) in enumerate(lines):
        if k + 1 < len(lines):
            b = min(b, lines[k + 1][0] - 0.01)
        out.append(f"{k + 1}\n{ts(a)} --> {ts(b)}\n{t}\n")
    return "\n".join(out)
