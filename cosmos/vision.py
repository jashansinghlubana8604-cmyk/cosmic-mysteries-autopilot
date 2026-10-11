"""Quick visual checks on candidate footage: people/faces and burned-in text are rejected (space b-roll should be
space, not staff photos, posters, presenters or captioned explainer videos)."""
from __future__ import annotations

import cv2
import numpy as np

_face = None
_hog = None


def _detectors():
    global _face, _hog
    if _face is None:
        _face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _face, _hog


def people_score(img: np.ndarray) -> float:
    """Fraction of the frame covered by detected faces / people (0 = none)."""
    face, hog = _detectors()
    h, w = img.shape[:2]
    k = 640 / max(w, h)
    small = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area = float(gray.shape[0] * gray.shape[1])
    cover = 0.0
    for (x, y, fw, fh) in face.detectMultiScale(gray, 1.1, 7, minSize=(22, 22)):
        cover += fw * fh / area * 4          # a face implies a person several times its size
    # HOG only on lit, terrestrial-looking scenes; on black space backgrounds it mistakes spacecraft for people
    if gray.std() > 25 and (gray > 60).mean() > 0.25 and (gray < 25).mean() < 0.15:
        rects, weights = hog.detectMultiScale(small, winStride=(6, 6), padding=(8, 8), scale=1.05)
        for (x, y, rw, rh), wt in zip(rects, np.ravel(weights) if len(weights) else []):
            if wt > 0.8:
                cover += rw * rh / area
    return min(1.0, cover)


def is_dark(img: np.ndarray) -> bool:
    small = cv2.resize(img, (480, max(1, int(img.shape[0] * 480 / img.shape[1]))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return bool(gray.mean() < 13 and np.percentile(gray, 99.5) < 110)


def layout_problem(img: np.ndarray) -> str:
    """Mosaic gaps (blank areas with straight edges) or panel grids / framed figures."""
    h, w = img.shape[:2]
    k = 480 / w
    small = cv2.resize(img, (480, max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    md = min(gray.shape)
    blank = (small.max(axis=2) <= 8).astype(np.uint8)
    if blank.mean() > 0.05:
        blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # ignore dark sky specks
        edges = cv2.Canny(blank * 255, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=int(0.30 * md), maxLineGap=3)
        if lines is not None and len(lines) >= 2:
            return "mosaic gap"
    bright = (gray > 140).astype(np.uint8)
    rows = (bright.mean(axis=1) > 0.6).sum()
    cols = (bright.mean(axis=0) > 0.6).sum()
    if gray.mean() < 120 and 1 <= rows <= 0.1 * gray.shape[0] and 1 <= cols <= 0.1 * gray.shape[1]:
        return "panel grid"
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=int(0.35 * md), maxLineGap=4)
    if lines is not None:
        hor = sum(1 for x1, y1, x2, y2 in lines[:, 0] if abs(y1 - y2) <= 1)
        ver = sum(1 for x1, y1, x2, y2 in lines[:, 0] if abs(x1 - x2) <= 1)
        if hor >= 2 and ver >= 2:
            return "boxes/frames"
    return ""


def text_score(img: np.ndarray) -> float:
    """Share of the frame covered by text-like line regions (captions, labels, posters)."""
    h, w = img.shape[:2]
    k = 960 / w
    gray = cv2.cvtColor(cv2.resize(img, (960, max(1, int(h * k))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.35:                   # textured everywhere (nebula, star field): not text
        _, bw = cv2.threshold(grad, 90, 255, cv2.THRESH_BINARY)
    conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(conn, connectivity=8)
    area = float(gray.shape[0] * gray.shape[1])
    total = 0.0
    lines = 0
    for i in range(1, n):
        x, y, cw, ch, a = stats[i]
        if 9 <= ch <= 70 and cw >= 3.2 * ch and cw >= 45:
            fill = a / float(cw * ch)
            if 0.35 <= fill <= 0.92:
                region = bw[y:y + ch, x:x + cw]
                # text rows have many vertical strokes: count transitions along the middle row
                mid = region[ch // 2]
                trans = int(np.count_nonzero(np.diff(mid.astype(np.int16)) != 0))
                if trans / max(1, cw) > 0.08:
                    total += cw * ch
                    lines += 1
    return total / area if lines >= 2 else 0.0


def _gray960(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.cvtColor(cv2.resize(img, (960, max(1, int(h * 960 / w))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def box_corners(gray: np.ndarray) -> int:
    """Corners of thin straight-edged boxes (insets, zoom frames, panel borders). A corner is where a horizontal and
    a vertical line both END; diffraction spikes cross through their middles, so stars don't count."""
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    m = (th > 45).astype(np.uint8)
    hor = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8))
    ver = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8))
    meet = cv2.dilate(hor, np.ones((5, 5), np.uint8)) & cv2.dilate(ver, np.ones((5, 5), np.uint8))
    n, _, _, cen = cv2.connectedComponentsWithStats(meet)
    H, W = gray.shape

    def has(mask, x, y):
        return mask[max(0, y - 2):min(H, y + 3), max(0, x - 2):min(W, x + 3)].any()

    found = 0
    for cx, cy in cen[1:].astype(int):
        if has(hor, cx - 12, cy) != has(hor, cx + 12, cy) and has(ver, cx, cy - 12) != has(ver, cx, cy + 12):
            found += 1
    return found


def plain_labels(gray: np.ndarray) -> int:
    """Bright text lines sitting on plain dark sky ('Mars Opposition', '2016'): short words text_score lets through."""
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.35:
        return 0
    conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(conn, connectivity=8)
    H, W = gray.shape
    hits = 0
    for x, y, cw, ch, a in stats[1:]:
        if not (14 <= ch <= 70 and cw >= 2.6 * ch and 0.45 <= a / float(cw * ch) <= 0.92):
            continue
        if np.count_nonzero(np.diff(bw[y + ch // 2, x:x + cw].astype(np.int16))) / cw <= 0.15:
            continue
        p = ch
        ring = gray[max(0, y - p):min(H, y + ch + p), max(0, x - p):min(W, x + cw + p)]
        inner = np.zeros(ring.shape, bool)
        inner[min(p, y):min(p, y) + ch, min(p, x):min(p, x) + cw] = True
        if (ring[~inner] < 25).mean() > 0.85 and gray[y:y + ch, x:x + cw].max() > 150:
            hits += 1
    return hits


def annotated(img: np.ndarray) -> str:
    """Press-release graphics a thumbnail must never show: inset boxes or caption labels."""
    gray = _gray960(img)
    if box_corners(gray) >= 2:
        return "boxes"
    if plain_labels(gray) >= 2:
        return "labels"
    return ""


def beauty(img: np.ndarray) -> float:
    """0..1: how striking a frame is (colourfulness, contrast, fine detail, a bright subject on dark)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (480, max(1, int(h * 480 / w))), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = (np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 110.0
    gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 70.0
    detail = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 7.0
    bright = (gray > 200).mean()
    subject = 1.0 - min(1.0, abs(bright - 0.06) * 6)
    dim = 0.0 if gray.mean() > 18 else (18 - gray.mean()) / 18
    dark_frac = (gray < 40).mean()
    space = 1.0 if 0.12 < dark_frac < 0.92 else 0.0          # the look of space: light against darkness
    sky = ((b > r + 25) & (b > 120) & (gray > 110)).mean()      # daylight sky / ground photos
    hsv = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2HSV)
    lit = hsv[..., 2] > 60
    hue_spread = float(np.std(hsv[..., 0][lit])) / 50.0 if lit.mean() > 0.02 else 0.0   # several colours, not one
    s = (0.18 * min(1.0, colorful) + 0.12 * min(1.0, hue_spread) + 0.18 * min(1.1, contrast)
         + 0.18 * min(1.1, detail) + 0.12 * subject + 0.25 * space - 0.3 * dim - 0.5 * min(1.0, sky * 3))
    return float(max(0.0, min(1.0, s)))


def verdict(img: np.ndarray) -> tuple[bool, str]:
    lp = layout_problem(img)
    if lp:
        return False, lp
    t = text_score(img)
    if t > 0.04:
        return False, f"text {t:.3f}"
    p = people_score(img)
    if p > 0.04:
        return False, f"people {p:.3f}"
    return True, f"ok t={t:.3f} p={p:.3f}"
