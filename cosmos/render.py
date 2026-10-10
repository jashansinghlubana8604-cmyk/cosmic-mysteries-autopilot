"""Picture: shot frames, camera moves, transitions, the space grade (bloom, vignette, grain), the warp-speed title
sequence, and the encoder (libass overlays burned in, H.264 + AAC).

Frames are composed in Python with sub-pixel precision (OpenCV) and piped into FFmpeg.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .media import FFMPEG, iter_frames, probe

log = logging.getLogger("cosmos.render")

W, H = 1920, 1080
FPS = 30
ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "assets" / "fonts"


class RenderError(Exception):
    pass


@dataclass
class Shot:
    kind: str                         # image | video | warp | black
    frames: int
    src: Path | None = None
    start: float = 0.0                # video: source start (s)
    speed: float = 1.0                # video: >1 = slow motion
    move: str = "push"
    strength: float = 1.0
    trans_in: str = "fade"            # cut | fade | zoom | dip | flash
    trans_len: int = 18
    dim: float = 0.0                  # constant darkening 0..1 (title / chapter cards)
    soft: float = 0.0                 # 0..1 background blur (cards)
    dims: list = field(default_factory=list)     # [(f0, f1, amount)] spotlight dims for on-screen text
    fade_out: int = 0                 # frames fading to black at the end
    seed: int = 0
    credit: str = ""
    label: str = ""


# ---- look -------------------------------------------------------------------------------------------------
def _grade_lut() -> np.ndarray:
    x = np.arange(256) / 255.0
    s = x * x * (3 - 2 * x)
    base = 0.55 * x + 0.45 * s                 # contrast S-curve: deep space blacks
    base = 0.008 + base * 0.99
    sh, hi = (1 - x) ** 2.5, x ** 2.2
    b = base + 0.035 * sh - 0.015 * hi        # cool shadows
    g = base + 0.010 * sh + 0.004 * hi
    r = base - 0.010 * sh + 0.025 * hi        # faintly warm highlights
    lut = np.stack([b, g, r], axis=-1).reshape(256, 1, 3)
    return np.clip(lut * 255, 0, 255).astype(np.uint8)


class Look:
    def __init__(self, seed: int = 1) -> None:
        self.lut = _grade_lut()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        m = 1.0 - 0.42 * np.clip((d - 0.30) / 0.70, 0, 1) ** 1.7
        self.vig = np.repeat((m * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        rng = np.random.default_rng(seed)
        self.grain = []
        for _ in range(6):
            n = rng.normal(0, 2.2, (H // 2, W // 2)).astype(np.float32)
            n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
            self.grain.append((cv2.merge([np.clip(n, 0, 255).astype(np.uint8)] * 3),
                               cv2.merge([np.clip(-n, 0, 255).astype(np.uint8)] * 3)))

    def apply(self, img: np.ndarray, gi: int) -> np.ndarray:
        img = cv2.LUT(img, self.lut)
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1.16, gray, -0.16, 0)                      # +16% saturation
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hi = cv2.subtract(small, (165, 165, 165, 0))
        hi = cv2.GaussianBlur(hi, (0, 0), 7)
        img = cv2.addWeighted(img, 1.0, cv2.resize(hi, (W, H), interpolation=cv2.INTER_LINEAR), 0.85, 0)  # bloom
        img = cv2.multiply(img, self.vig, scale=1 / 255)
        pos, neg = self.grain[(gi // 2) % len(self.grain)]
        return cv2.subtract(cv2.add(img, pos), neg)


# ---- camera ------------------------------------------------------------------------------------------------
def ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t)))


def view(src: np.ndarray, cx: float, cy: float, scale: float, rot_deg: float = 0.0) -> np.ndarray:
    """Output W x H looking at (cx, cy) in source pixels; `scale` = output px per source px."""
    th = math.radians(rot_deg)
    c, s = math.cos(th) / scale, math.sin(th) / scale
    a = np.array([[c, -s], [s, c]], np.float64)
    b = np.array([cx, cy]) - a @ np.array([W / 2, H / 2])
    m = np.hstack([a, b[:, None]]).astype(np.float32)
    return cv2.warpAffine(src, m, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT101)


def _focus(img: np.ndarray) -> tuple[float, float]:
    """Brightness-weighted centre of interest (space subjects are bright on dark), pulled toward the middle."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
    thr = np.percentile(g, 92)
    wts = np.clip(g - thr, 0, None) ** 1.5
    if wts.sum() < 1e-3:
        return w / 2, h / 2
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    fx = (xx * wts).sum() / wts.sum() / g.shape[1] * w
    fy = (yy * wts).sum() / wts.sum() / g.shape[0] * h
    return 0.45 * w / 2 + 0.55 * fx, 0.45 * h / 2 + 0.55 * fy


def load_image(path: Path) -> np.ndarray:
    """Decode at a reduced size when the file is huge (archive images can exceed a gigapixel)."""
    flag = cv2.IMREAD_COLOR
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            iw, ih = im.size
        biggest = max(iw, ih)
        for f, fl in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if biggest / f >= 3600:
                flag = fl
                break
    except Exception:  # noqa: BLE001
        pass
    img = cv2.imread(str(path), flag)
    if img is None:
        raise RenderError(f"cannot read {path}")
    h, w = img.shape[:2]
    cover = max(W / w, H / h)
    if cover < 1 / 1.7:                       # keep ~1.7x headroom for zooms, no more (memory, speed)
        k = cover * 1.7
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img


class ImageCam:
    def __init__(self, img: np.ndarray, move: str, strength: float, n: int, seed: int) -> None:
        self.img = img
        h, w = img.shape[:2]
        self.base = max(W / w, H / h)
        self.fx, self.fy = _focus(img)
        self.w, self.h, self.n = w, h, n
        self.move, self.k = move, strength
        rng = np.random.default_rng(seed)
        self.rot = rng.uniform(-0.7, 0.7) if move == "float" else 0.0

    def _clamp(self, cx, cy, z):
        hx, hy = W / (2 * self.base * z), H / (2 * self.base * z)
        return min(max(cx, hx), self.w - hx), min(max(cy, hy), self.h - hy)

    def frame(self, i: int) -> np.ndarray:
        p = ease(i / max(1, self.n - 1))
        k = self.k
        mx, my = self.w / 2, self.h / 2
        if self.move == "push":
            z = 1.02 + 0.13 * k * p
            cx, cy = mx + (self.fx - mx) * p, my + (self.fy - my) * p
        elif self.move == "pull":
            z = 1.15 * (1 + 0.0 * k) - 0.13 * k * p
            cx, cy = self.fx + (mx - self.fx) * p, self.fy + (my - self.fy) * p
        elif self.move in ("pan_l", "pan_r"):
            z = 1.12 + 0.03 * k
            hx = W / (2 * self.base * z)
            slack = max(0.0, self.w / 2 - hx)
            d = slack * 0.8 * k * (1 if self.move == "pan_r" else -1)
            cx, cy = mx - d / 2 + d * p, my + (self.fy - my) * 0.5
        elif self.move in ("tilt_u", "tilt_d"):
            z = 1.12 + 0.03 * k
            hy = H / (2 * self.base * z)
            slack = max(0.0, self.h / 2 - hy)
            d = slack * 0.8 * k * (1 if self.move == "tilt_d" else -1)
            cx, cy = mx + (self.fx - mx) * 0.5, my - d / 2 + d * p
        else:  # float: slow push with a little roll
            z = 1.08 + 0.08 * k * p
            cx, cy = mx + (self.fx - mx) * 0.6, my + (self.fy - my) * 0.6
            cx, cy = self._clamp(cx, cy, z)
            return view(self.img, cx, cy, self.base * z, self.rot * (2 * p - 1))
        cx, cy = self._clamp(cx, cy, z)
        return view(self.img, cx, cy, self.base * z)


def _video_frames(shot: Shot, n: int):
    info = probe(shot.src)
    sw, sh = info["width"], info["height"]
    cover = max(W / sw, H / sh)
    dw, dh = int(round(sw * cover / 2)) * 2, int(round(sh * cover / 2)) * 2
    dw, dh = max(dw, W), max(dh, H)
    prefix = f"setpts={shot.speed:.4f}*(PTS-STARTPTS),fps={FPS},"
    count, last = 0, None
    for f in iter_frames(shot.src, dw, dh, start=shot.start, duration=n / FPS / shot.speed + 0.6, vf_prefix=prefix):
        if count >= n:
            break
        p = ease(count / max(1, n - 1))
        z = 1.0 + 0.05 * shot.strength * (p if shot.move != "pull" else 1 - p)
        img = view(f, dw / 2, dh / 2, z) if (dw, dh) != (W, H) or z != 1.0 else f
        last = img
        count += 1
        yield img
    if last is None:
        raise RenderError(f"could not decode {shot.src} @ {shot.start:.1f}s")
    while count < n:
        count += 1
        yield last


class Warp:
    """Procedural warp-speed starfield (title sequences)."""

    def __init__(self, n_frames: int, seed: int, drift: bool = False) -> None:
        rng = np.random.default_rng(seed)
        self.drift = drift
        self.N = 2200 if drift else 1400
        self.x = rng.uniform(-1.6, 1.6, self.N)
        self.y = rng.uniform(-0.9, 0.9, self.N)
        self.z = rng.uniform(0.15, 4.0, self.N)
        self.col = rng.uniform(0.75, 1.0, (self.N, 1)) * np.array([[255, 235, 215]])
        self.n = n_frames
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
        d = np.sqrt(((xx - W / 8) / (W / 8)) ** 2 + ((yy - H / 8) / (H / 8)) ** 2)
        haze = np.exp(-d * 1.8)[:, :, None] * np.array([[[90, 40, 20]]], np.float32)
        if drift:   # soft two-colour nebula haze
            n1 = cv2.GaussianBlur(rng.random((H // 16, W // 16)).astype(np.float32), (0, 0), 3)
            n2 = cv2.GaussianBlur(rng.random((H // 16, W // 16)).astype(np.float32), (0, 0), 4)
            n1 = cv2.resize((n1 - n1.min()) / (np.ptp(n1) + 1e-6), (W // 4, H // 4))
            n2 = cv2.resize((n2 - n2.min()) / (np.ptp(n2) + 1e-6), (W // 4, H // 4))
            haze = (n1[..., None] ** 2.5 * np.array([[[120, 50, 70]]], np.float32)
                    + n2[..., None] ** 3 * np.array([[[40, 60, 130]]], np.float32)) * 0.9
        self.haze = cv2.resize(haze, (W, H)).astype(np.float32)

    def frame(self, i: int) -> np.ndarray:
        p = i / max(1, self.n - 1)
        speed = (0.0035 + 0.004 * p) if self.drift else (0.003 + 0.17 * p ** 2.6)
        z0 = self.z.copy()
        self.z -= speed
        wrap = self.z < 0.05
        self.z[wrap] += 4.0
        z0[wrap] = self.z[wrap] + speed
        f = 520.0
        x1, y1 = self.x / self.z * f + W / 2, self.y / self.z * f + H / 2
        x0, y0 = self.x / z0 * f + W / 2, self.y / z0 * f + H / 2
        canvas = (self.haze * ((0.55 + 0.45 * min(1.0, p * 2)) if self.drift else (0.4 + 0.8 * p))).astype(np.uint8)
        b = np.clip(1.4 / self.z, 0.08, 1.0)
        for k in range(self.N):
            if not (-50 < x1[k] < W + 50 and -50 < y1[k] < H + 50):
                continue
            c = tuple(float(v) * b[k] for v in self.col[k])
            th = 1 if b[k] < 0.5 else 2
            p0, p1 = (int(x0[k]), int(y0[k])), (int(x1[k]), int(y1[k]))
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 2:
                cv2.circle(canvas, p1, th, c, -1, cv2.LINE_AA)
            else:
                cv2.line(canvas, p0, p1, c, th, cv2.LINE_AA)
        glow = cv2.GaussianBlur(cv2.resize(canvas, (W // 4, H // 4)), (0, 0), 4)
        return cv2.addWeighted(canvas, 1.0, cv2.resize(glow, (W, H)), 1.2, 0)


def _soften(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    small = cv2.resize(img, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    soft = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1 - amount, soft, amount, 0)


def _zoom_center(img: np.ndarray, z: float, blur: int = 0) -> np.ndarray:
    if z != 1.0:
        img = view(img, W / 2, H / 2, z)
    if blur > 2:
        small = cv2.resize(img, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
        img = cv2.resize(cv2.blur(small, (blur // 3 + 1, blur // 3 + 1)), (W, H))
    return img


def _exposure_lut(sample: np.ndarray) -> np.ndarray | None:
    """Gamma lift for dim shots (deep-field images, dark animations): brighten shadows and mid-tones without
    clipping the stars. None when the shot is already bright enough."""
    g = cv2.cvtColor(cv2.resize(sample, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    if mean >= 38:
        return None
    gamma = float(np.clip(1.0 + (38 - mean) / 38 * 0.75, 1.0, 1.7))
    x = np.arange(256) / 255.0
    return np.clip((x ** (1 / gamma)) * 255, 0, 255).astype(np.uint8)


def shot_frames(shot: Shot, n: int):
    """Yield n composed (pre-look) frames of a shot, auto-exposed."""
    lut = None
    for i, f in enumerate(_shot_frames_raw(shot, n)):
        if i == 0 and shot.kind in ("image", "video"):
            lut = _exposure_lut(f)
        yield cv2.LUT(f, lut) if lut is not None else f


def _shot_frames_raw(shot: Shot, n: int):
    if shot.kind == "black":
        z = np.zeros((H, W, 3), np.uint8)
        for _ in range(n):
            yield z
        return
    if shot.kind == "warp":
        wp = Warp(n, shot.seed, drift=shot.move == "drift")
        for i in range(n):
            yield wp.frame(i)
        return
    if shot.kind == "image":
        cam = ImageCam(load_image(shot.src), shot.move, shot.strength, n, shot.seed)
        for i in range(n):
            yield cam.frame(i)
        return
    yield from _video_frames(shot, n)


# ---- encode -------------------------------------------------------------------------------------------------
def render(shots: list[Shot], ass: Path, audio: Path, out: Path, work: Path, seed: int = 1,
           crf: int = 19, preset: str = "medium", progress=None) -> None:
    total = sum(s.frames for s in shots)
    fonts = work / "fonts"
    if fonts.exists():
        shutil.rmtree(fonts)
    shutil.copytree(FONTS_DIR, fonts)
    shutil.copy(ass, work / "overlay.ass")
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", str(audio.resolve()),
           "-filter_complex", "[0:v]ass=overlay.ass:fontsdir=fonts,format=yuv420p[v]",
           "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-profile:v", "high", "-tune", "film", "-g", str(FPS * 2), "-bf", "2",
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
           "-t", f"{total / FPS:.3f}", str(out.resolve())]
    errlog = open(work / "ffmpeg.log", "wb")
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog, cwd=work)
    look = Look(seed)
    gi = 0
    tail: list[np.ndarray] = []
    try:
        for si, shot in enumerate(shots):
            nxt = shots[si + 1] if si + 1 < len(shots) else None
            overlap = nxt.trans_len if nxt and nxt.trans_in in ("fade", "zoom", "flash") else 0
            n = shot.frames + overlap
            new_tail = []
            L = shot.trans_len
            for fi, img in enumerate(shot_frames(shot, n)):
                if fi >= shot.frames:
                    new_tail.append(img)
                    continue
                if shot.dim or shot.soft:
                    img = _soften(img, shot.soft)
                    if shot.dim:
                        img = cv2.convertScaleAbs(img, alpha=1 - shot.dim)
                flash = 0.0
                if fi < L and tail and shot.trans_in in ("fade", "zoom", "flash"):
                    a = (fi + 1) / (L + 1)
                    prev = tail[min(fi, len(tail) - 1)]
                    if shot.trans_in == "zoom":
                        e = ease(a)
                        prev = _zoom_center(prev, 1 + 0.35 * e, int(60 * e))
                        img = _zoom_center(img, 1.25 - 0.25 * e, int(60 * (1 - e)))
                        flash = 0.35 * math.sin(math.pi * a)
                        img = cv2.addWeighted(prev, 1 - e, img, e, 0)
                    elif shot.trans_in == "flash":
                        flash = 0.8 * (1 - a) ** 1.5
                        img = cv2.addWeighted(prev, 1 - a, img, a, 0)
                    else:
                        e = ease(a)
                        img = cv2.addWeighted(prev, 1 - e, img, e, 0)
                for f0, f1, amt in shot.dims:
                    if f0 <= fi < f1:
                        r = max(0.0, min(1.0, (fi - f0) / 8, (f1 - fi) / 12))
                        img = cv2.convertScaleAbs(img, alpha=1 - amt * r)
                img = look.apply(img, gi)
                if flash > 0:
                    img = cv2.addWeighted(img, 1 - flash, np.full_like(img, (255, 240, 225)), flash, 0)
                fade = 1.0
                if shot.trans_in == "dip" and fi < L:
                    fade = min(fade, (fi + 1) / L)
                left = shot.frames - 1 - fi
                if nxt is not None and nxt.trans_in == "dip" and left < nxt.trans_len:
                    fade = min(fade, left / nxt.trans_len)
                if shot.fade_out and left < shot.fade_out:
                    fade = min(fade, left / shot.fade_out)
                if fade < 1.0:
                    img = cv2.convertScaleAbs(img, alpha=ease(max(0.0, fade)))
                enc.stdin.write(np.ascontiguousarray(img).tobytes())
                gi += 1
                if progress and gi % 300 == 0:
                    progress(gi, total)
            tail = new_tail
    except (BrokenPipeError, OSError) as e:
        enc.kill()
        errlog.close()
        raise RenderError(f"encoder stopped: {(work / 'ffmpeg.log').read_text(errors='ignore')[-800:]} {e}")
    enc.stdin.close()
    code = enc.wait()
    errlog.close()
    if code != 0:
        raise RenderError(f"ffmpeg failed: {(work / 'ffmpeg.log').read_text(errors='ignore')[-800:]}")
