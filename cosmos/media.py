"""FFmpeg helpers: probing, scene-cut detection, visual fingerprints, AI proxy files."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def probe(path: Path) -> dict:
    out = run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]).stdout
    data = json.loads(out)
    v = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    duration = float(data["format"].get("duration") or v.get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return {"width": w, "height": h, "fps": fps if 1 < fps < 121 else 30.0, "duration": duration, "has_audio": has_audio}


def iter_frames(path: Path, width: int, height: int, *, gray: bool = False, start: float = 0.0,
                duration: float | None = None, vf_prefix: str = ""):
    """Yield decoded frames (uint8 numpy) scaled to width x height."""
    pix, ch = ("gray", 1) if gray else ("bgr24", 3)
    vf = f"{vf_prefix}scale={width}:{height}:flags=area"
    cmd = [FFMPEG, "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", vf, "-pix_fmt", pix, "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * ch
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            arr = np.frombuffer(buf, np.uint8)
            yield arr.reshape(height, width) if gray else arr.reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def scene_cuts(path: Path, fps: float) -> list[float]:
    """Timestamps (seconds) of hard cuts inside the source, so our shots never start or end on a flash frame."""
    diffs = []
    prev = None
    for f in iter_frames(path, 96, 96, gray=True):
        f = f.astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(f - prev))))
        prev = f
    if len(diffs) < 3:
        return []
    d = np.array(diffs)
    thresh = max(22.0, 5.0 * float(np.median(d[1:])))
    cuts = []
    for i in range(1, len(d) - 1):
        if d[i] > thresh and d[i] >= d[i - 1] and d[i] >= d[i + 1]:
            cuts.append(i / fps)
    return cuts


def _frame_at(path: Path, t: float, w: int, h: int) -> np.ndarray | None:
    try:
        out = run([FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                   "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "gray", "-f", "rawvideo", "-"]).stdout
    except subprocess.CalledProcessError:
        return None
    if len(out) < w * h:
        return None
    return np.frombuffer(out[: w * h], np.uint8).reshape(h, w)


def fingerprint(path: Path, duration: float, samples: int = 8) -> list[int]:
    """Difference-hashes of frames spread through the clip; flat/black frames are skipped."""
    hashes = []
    for i in range(samples):
        t = duration * (i + 0.5) / samples
        f = _frame_at(path, t, 32, 32)
        if f is None or float(f.std()) < 8.0:
            continue
        small = cv2.resize(f, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        hashes.append(int("".join("1" if b else "0" for b in bits), 2))
    return hashes


def find_duplicate(fp: list[int], store: dict[str, list[int]], max_dist: int = 10) -> str | None:
    """Key of an already-used clip that shows the same footage, if any."""
    if len(fp) < 3:
        return None
    need = max(3, len(fp) // 2)
    for key, other in store.items():
        if not other:
            continue
        hits = sum(1 for h in fp if min(bin(h ^ o).count("1") for o in other) <= max_dist)
        if hits >= need:
            return key
    return None


def make_proxy(src: Path, dst: Path, max_seconds: float) -> None:
    """Small copy of the clip for the AI to watch (fast upload, same timeline)."""
    run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-t", f"{max_seconds:.2f}",
         "-vf", "scale='if(gt(iw,ih),-2,480)':'if(gt(iw,ih),480,-2)'", "-r", "15",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "64k", "-ac", "1",
         str(dst)])
