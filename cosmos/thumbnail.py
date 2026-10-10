"""Thumbnails (1280x720) built to create awe: the most striking on-topic space image, framed big and close,
graded like a film still (deep blacks, glowing highlights), with 2-4 words of heavy, confident type.

Text comes from the script's `thumb:` line, e.g. `WHO SENT | *THIS?*` (| = new line, *word* = gold accent).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import vision
from .render import _focus

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "BebasNeue-Regular.ttf"
TW, TH = 1280, 720
GOLD = (255, 205, 92)
ACCENT = (255, 204, 40)
WHITE = (250, 250, 248)


def striking(paths: list[Path], top: int = 5) -> Path | None:
    """Of the first `top` candidates (already ranked by relevance), the most beautiful clean one: no labels,
    no diagrams, landscape-friendly."""
    best, best_s = None, -1e9
    for rank_i, p in enumerate(paths[:top]):
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w < 320 or h < 180:
            continue
        big = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
        if vision.text_score(big) > 0.004:                      # even small labels ruin a thumbnail
            continue
        s = vision.beauty(img) - 0.09 * rank_i + (0.05 if w >= h else -0.1)
        if s > best_s:
            best, best_s = p, s
    return best


FONTS = ROOT / "assets" / "fonts"


def _grade(arr: np.ndarray, style: str) -> np.ndarray:
    """BGR float 0..1 -> film-still look: S-curve, rich colour, glowing highlights, darkness where the words go."""
    x = np.clip(arr, 0, 1)
    x = x * x * (3 - 2 * x) * 0.5 + x * 0.5                         # contrast
    gray = x.mean(axis=2, keepdims=True)
    x = np.clip(gray + (x - gray) * (1.45 if style == "impact" else 1.3), 0, 1)   # saturation
    hi = np.clip(x - 0.5, 0, 1)
    glow = cv2.GaussianBlur(hi, (0, 0), 22) + cv2.GaussianBlur(hi, (0, 0), 7) * 0.7
    x = 1 - (1 - x) * (1 - np.clip(glow * 1.2, 0, 1))                # screen-blend bloom: the subject glows
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    cx, cy = (0.62, 0.5) if style == "impact" else (0.5, 0.42 if style == "cinematic" else 0.6)
    d = np.sqrt(((xx - TW * cx) / (TW * 0.7)) ** 2 + ((yy - TH * cy) / (TH * 0.75)) ** 2)
    vig = 1 - 0.6 * np.clip(d - 0.3, 0, 1) ** 1.3
    if style == "cinematic":                                         # darker band under the title
        band = 1 - 0.55 * np.exp(-((yy - TH * 0.74) / (TH * 0.16)) ** 2)
    elif style == "impact":                                          # darker left side for the stacked words
        band = 1 - 0.62 * np.clip(1 - xx / (TW * 0.62), 0, 1) ** 1.3
    else:                                                            # darker top for the headline
        band = 1 - 0.6 * np.clip(1 - yy / (TH * 0.38), 0, 1) ** 1.5
    return np.clip(x * vig[..., None] * band[..., None], 0, 1)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _runs(line: str) -> list[tuple[str, bool]]:
    """'TOO BIG TO *EXIST*' -> [('TOO BIG TO ', False), ('EXIST', True)]"""
    return [(chunk.upper(), i % 2 == 1) for i, chunk in enumerate(line.split("*")) if chunk]


def _plain(runs: list[tuple[str, bool]]) -> str:
    return "".join(t for t, _ in runs)


def _tracked_width(d: ImageDraw.ImageDraw, text: str, font, track: float) -> float:
    return sum(d.textlength(ch, font=font) for ch in text) + track * max(0, len(text) - 1)


def _draw_runs(d: ImageDraw.ImageDraw, runs: list[tuple[str, bool]], x: float, y: float, font, track: float,
               fill, accent: bool = True, stroke: int = 0, stroke_fill=(0, 0, 0, 255), align: str = "center") -> None:
    """One line of runs with letter tracking; accent runs in gold unless `accent` is False (shadows, glows).
    x is the centre (align="center") or the left edge."""
    if align == "center":
        x -= _tracked_width(d, _plain(runs), font, track) / 2
    for t, acc in runs:
        col = ACCENT + (fill[3],) if acc and accent else fill
        for ch in t:
            d.text((x, y), ch, font=font, fill=col, stroke_width=stroke, stroke_fill=stroke_fill)
            x += d.textlength(ch, font=font) + track


def _layer(img: Image.Image, painter, blur: float = 0) -> Image.Image:
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    painter(ImageDraw.Draw(lay))
    if blur:
        lay = lay.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(img, lay)


def _frame(image: Path, style: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop/zoom so one subject dominates: large, on darkness, leaving room for the words.
    Returns the frame and where the subject ended up inside it."""
    src = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if src is None:
        src = cv2.imread(str(image), cv2.IMREAD_REDUCED_COLOR_4)
    if src is None:
        raise ValueError(f"cannot read {image}")
    h, w = src.shape[:2]
    scale = max(TW / w, TH / h) * 1.05
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(src, (320, max(1, int(320 * h / w)))), cv2.COLOR_BGR2GRAY), (0, 0), 4)
    ys, xs = np.where(g > max(40, np.percentile(g, 96)))
    if len(xs) > 20:
        bw = (np.percentile(xs, 95) - np.percentile(xs, 5) + 1) / 320 * w * scale
        if bw < TW * 0.40:
            scale *= min(2.2, (TW * 0.52) / max(bw, 1))
    src = cv2.resize(src, (int(w * scale) + 2, int(h * scale) + 2), interpolation=cv2.INTER_AREA)
    fx, fy = _focus(src)
    tx, ty = {"cinematic": (0.5, 0.40), "impact": (0.68, 0.5)}.get(style, (0.5, 0.60))   # room for the words
    x0 = int(min(max(fx - TW * tx, 0), src.shape[1] - TW))
    y0 = int(min(max(fy - TH * ty, 0), src.shape[0] - TH))
    return src[y0:y0 + TH, x0:x0 + TW].astype(np.float32) / 255, (fx - x0, fy - y0)


def _circle(img: Image.Image, cx: float, cy: float, r: float) -> Image.Image:
    """The red ring the outlier thumbnails put around the strange detail."""
    box = (cx - r, cy - r, cx + r, cy + r)
    img = _layer(img, lambda d: d.ellipse(box, outline=(0, 0, 0, 200), width=22), 6)
    img = _layer(img, lambda d: d.ellipse(box, outline=(255, 40, 40, 160), width=18), 10)
    return _layer(img, lambda d: d.ellipse(box, outline=(235, 28, 28, 255), width=11))


def make(image: Path, text: str, out: Path, badge: str = "", style: str = "cinematic", mark: str = "") -> Path:
    """style "cinematic": wide-spaced title across the lower third + small subtitle (END OF REALITY / TON-618).
    style "headline": very heavy words across the top, subject below (THIS IS EUROPA).
    style "impact": 2-3 stacked heavy lines on the left, subject glowing on the right (THE SKY / ANSWERED).
    `*word*` is drawn in gold; mark "circle" rings the subject in red."""
    frame, (sx, sy) = _frame(image, style)
    graded = (_grade(frame, style) * 255).astype(np.uint8)
    img = Image.fromarray(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB)).convert("RGBA")
    if mark == "circle":
        left = TW * 0.66 if style == "impact" else 140                  # never over the words
        top = TH * 0.42 if style == "headline" else 140
        img = _circle(img, min(max(sx, left), TW - 140), min(max(sy, top), TH - 140), 118)
    lines = [t.strip() for t in text.split("|") if t.strip()]
    main = _runs(lines[0]) if lines else []
    sub = _runs(lines[1]) if len(lines) > 1 else []
    d0 = ImageDraw.Draw(img)
    if style == "impact":
        rows = [_runs(t) for t in lines[:3]]
        size = 190
        while size > 80:
            f = _font("BebasNeue-Regular.ttf", size)
            if max(_tracked_width(d0, _plain(r), f, size * 0.02) for r in rows) <= TW * 0.56 and \
                    size * 0.92 * len(rows) <= TH * 0.82:
                break
            size -= 4
        f = _font("BebasNeue-Regular.ttf", size)
        step = size * 0.92
        y0 = (TH - step * len(rows)) / 2 - size * 0.06
        for i, r in enumerate(rows):
            y = y0 + i * step
            img = _layer(img, lambda d, r=r, y=y: _draw_runs(d, r, 64, y + 10, f, size * 0.02, (0, 0, 0, 235),
                                                             accent=False, align="left"), 18)
            img = _layer(img, lambda d, r=r, y=y: _draw_runs(d, r, 64, y, f, size * 0.02, (252, 252, 250, 255),
                                                             stroke=3, stroke_fill=(10, 10, 10, 255), align="left"))
    elif style == "headline":
        size = 150
        while size > 70:
            f = _font("Montserrat-Black.ttf", size)
            if _tracked_width(d0, _plain(main), f, size * 0.01) <= TW * 0.92:
                break
            size -= 4
        f = _font("Montserrat-Black.ttf", size)
        y = 40
        img = _layer(img, lambda d: _draw_runs(d, main, TW / 2, y + 9, f, size * 0.01, (0, 0, 0, 235), accent=False), 16)
        img = _layer(img, lambda d: _draw_runs(d, main, TW / 2, y, f, size * 0.01, (255, 255, 255, 255),
                                               stroke=3, stroke_fill=(12, 12, 12, 255)))
        if sub:
            fs = _font("Montserrat-ExtraBold.ttf", 46)
            ys = y + size * 1.18
            img = _layer(img, lambda d: _draw_runs(d, sub, TW / 2, ys + 4, fs, 10, (0, 0, 0, 220), accent=False), 8)
            img = _layer(img, lambda d: _draw_runs(d, sub, TW / 2, ys, fs, 10, (236, 236, 236, 255)))
    else:
        size = 118
        track_k = 0.22
        while size > 60:
            f = _font("Montserrat-SemiBold.ttf", size)
            if _tracked_width(d0, _plain(main), f, size * track_k) <= TW * 0.86:
                break
            size -= 3
        f = _font("Montserrat-SemiBold.ttf", size)
        y = int(TH * 0.66) - size // 2
        tr = size * track_k
        img = _layer(img, lambda d: _draw_runs(d, main, TW / 2, y + 6, f, tr, (0, 0, 0, 220), accent=False), 14)
        img = _layer(img, lambda d: _draw_runs(d, main, TW / 2, y, f, tr, (255, 255, 255, 120), accent=False), 10)
        img = _layer(img, lambda d: _draw_runs(d, main, TW / 2, y, f, tr, (255, 255, 255, 255)))
        if sub:
            fs = _font("Montserrat-Medium.ttf", 38)
            ys = y + int(size * 1.25)
            img = _layer(img, lambda d: _draw_runs(d, sub, TW / 2, ys + 3, fs, 22, (0, 0, 0, 200), accent=False), 8)
            img = _layer(img, lambda d: _draw_runs(d, sub, TW / 2, ys, fs, 22, (225, 225, 225, 255)))
    if badge:
        d = ImageDraw.Draw(img)
        bf = _font("Montserrat-ExtraBold.ttf", 34)
        bw = d.textlength(badge.upper(), font=bf)
        bx = TW - 40 - bw - 30 if style == "impact" else 40
        d.rounded_rectangle((bx, 36, bx + bw + 30, 36 + 54), radius=8, fill=(205, 30, 30, 240))
        d.text((bx + 15, 43), badge.upper(), font=bf, fill=(255, 255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, "JPEG", quality=93, optimize=True)
    return out


def candidates(ep, fx: Any) -> list[Path]:
    """Thumbnail images for an episode, best first: exact subject first, then awe, never ground clutter."""
    from .edit import AWE, EARTHLY
    qs = [q if q.startswith(("i:", "c:")) else f"i: {q.split(':', 1)[-1].strip()}" for q in ep.thumb_queries]
    fx.search_all(qs)
    cands = fx.candidates(qs, per=10)
    primary = {c.key for c in fx.search(qs[0])[:10]}          # the first search names the exact subject
    fx.fetch(cands[:18])
    ranked = []
    for c in cands:
        m = fx.media.get(c.key)
        if m and m.kind == "image":
            s = c.score + 0.25 * m.beauty + (0.2 if AWE.search(c.title) else 0) + (0.4 if c.key in primary else 0)
            if re.search(r"transporter|vehicle|truck|\bcar\b|road|people|visitor|crane", c.title, re.I):
                s -= 1.0
            if EARTHLY.search(c.title) and not EARTHLY.search(" ".join(ep.thumb_queries)):
                s -= 0.5
            ranked.append((s, m.path))
    ranked.sort(key=lambda x: -x[0])
    return [p for _, p in ranked]


USED = ROOT / "state" / "thumb_images.json"


def _used() -> dict[str, str]:
    import json
    return json.loads(USED.read_text(encoding="utf-8")) if USED.exists() else {}


def remember(ep_key: str, image: Path) -> None:
    """Record which image an episode's thumbnail uses, so no two episodes ever share one."""
    import json
    used = _used()
    used[ep_key] = image.name
    USED.parent.mkdir(parents=True, exist_ok=True)
    USED.write_text(json.dumps(used, indent=1, sort_keys=True), encoding="utf-8")


def pick(ep, cache: Path) -> Path | None:
    from .edit import Footage
    taken = {name for key, name in _used().items() if key != ep.key}
    best = striking([p for p in candidates(ep, Footage(cache, 1)) if p.name not in taken])
    if best:
        remember(ep.key, best)
    return best
