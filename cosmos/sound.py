"""Sound design and the final mix.

- Cinematic effects are synthesized (deterministic from a seed): sub booms, risers, whooshes, braams, shimmers,
  text ticks and a low drone bed for cold opens. No sample packs needed.
- Music: Kevin MacLeod tracks (CC BY 4.0, credited automatically), chosen per section mood, crossfaded,
  ducked under the narrator and swelling in the gaps.
- Master: stereo, -14 LUFS integrated (YouTube's reference), -1.5 dBTP.
"""
from __future__ import annotations

import json
import random
import subprocess
import wave
from pathlib import Path

import numpy as np

from .media import FFMPEG

SR = 48000
ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "assets" / "music"

VOICE_CHAIN = (
    "highpass=f=70,"
    "equalizer=f=120:t=q:w=1.0:g=2.0,"        # chest
    "equalizer=f=300:t=q:w=1.2:g=-1.5,"       # less boxy
    "equalizer=f=3500:t=q:w=1.4:g=2.0,"       # presence
    "equalizer=f=9000:t=q:w=1.5:g=1.0,"       # air
    "acompressor=threshold=-20dB:ratio=3:attack=5:release=90:makeup=2.5,"
    "alimiter=limit=0.95"
)


# ---- io --------------------------------------------------------------------------------------------------
def load(path: Path, channels: int = 1, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", str(channels), "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    return a.reshape(-1, channels) if channels > 1 else a


def save_stereo(arr: np.ndarray, path: Path) -> None:
    pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def db(x: float) -> float:
    return 10 ** (x / 20)


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


# ---- dsp helpers ------------------------------------------------------------------------------------------
def _fft_filter(sig: np.ndarray, lo: float = 0.0, hi: float = SR / 2, slope: float = 1.0) -> np.ndarray:
    """Zero-phase band-limit with soft (raised-cosine-ish) edges."""
    n = len(sig)
    size = 1 << int(np.ceil(np.log2(max(2, n))))
    spec = np.fft.rfft(sig, size)
    f = np.fft.rfftfreq(size, 1 / SR)
    g = np.ones_like(f)
    if lo > 0:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (4 * slope))
    if hi < SR / 2:
        g *= 1 / (1 + (f / hi) ** (4 * slope))
    return np.fft.irfft(spec * g, size)[:n]


def _sweep_bandpass(noise: np.ndarray, f0: float, f1: float, q: float) -> np.ndarray:
    out = np.zeros_like(noise)
    n = len(noise)
    x1 = x2 = y1 = y2 = 0.0
    block = 64
    for s in range(0, n, block):
        f = f0 * (f1 / f0) ** (s / n)
        w0 = 2 * np.pi * f / SR
        alpha = np.sin(w0) / (2 * q)
        a0 = 1 + alpha
        b0, b2, a1, a2 = alpha / a0, -alpha / a0, -2 * np.cos(w0) / a0, (1 - alpha) / a0
        for i in range(s, min(n, s + block)):
            x0 = noise[i]
            y0 = b0 * x0 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
    return out


def reverb(sig: np.ndarray, seconds: float = 3.0, wet: float = 0.4, seed: int = 3, dark: float = 6000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 6.5, n))
    ir = _fft_filter(ir, 80, dark)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    size = 1 << int(np.ceil(np.log2(len(sig) + n)))
    tail = np.fft.irfft(np.fft.rfft(sig, size) * np.fft.rfft(ir, size), size)[: len(sig) + n]
    dry = np.pad(sig, (0, n))
    return (1 - wet) * dry + wet * tail * 0.7


def _norm(sig: np.ndarray, peak: float = 0.8) -> np.ndarray:
    return (sig / (np.abs(sig).max() + 1e-9) * peak).astype(np.float32)


# ---- effects ------------------------------------------------------------------------------------------------
def boom(seed: int, size: float = 1.0) -> np.ndarray:
    """Cinematic sub hit: pitched-down sine + body thump + noise crack, with a long dark tail."""
    rng = np.random.default_rng(seed)
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    freq = 34 + 95 * np.exp(-t * 7)
    sub = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 1.6)
    body = np.sin(2 * np.pi * np.cumsum(70 + 160 * np.exp(-t * 25)) / SR) * np.exp(-t * 9)
    crack = _fft_filter(rng.standard_normal(n) * np.exp(-t * 35), 200, 5000)
    sig = 1.0 * sub + 0.5 * body + 0.35 * crack * size
    sig = reverb(sig, 3.5, 0.35, seed, 3000)
    return _norm(sig, 0.9)


def whoosh(seed: int, length: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    noise = rng.standard_normal(n)
    sig = _sweep_bandpass(noise, rng.uniform(250, 450), rng.uniform(3000, 5000), 1.2)
    t = np.linspace(0, 1, n)
    peak = 0.6
    env = np.where(t < peak, (t / peak) ** 2.5, np.exp(-(t - peak) * 8))
    return _norm(reverb(sig * env, 1.2, 0.25, seed, 7000), 0.7)


def riser(seed: int, length: float = 3.0) -> np.ndarray:
    """Tension build: rising filtered noise + rising tone cluster, ends abruptly (the hit lands after it)."""
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    t = np.arange(n) / SR
    p = t / length
    noise = _sweep_bandpass(rng.standard_normal(n), 300, 6000, 0.9)
    tone = np.zeros(n)
    for k, base in enumerate((110, 164.8, 220, 329.6)):
        f = base * (1 + 1.0 * p ** 2)
        tone += np.sin(2 * np.pi * np.cumsum(f) / SR + k) / (k + 1)
    trem = 0.75 + 0.25 * np.sin(2 * np.pi * np.cumsum(3 + 14 * p ** 2) / SR)
    env = p ** 2.2
    sig = (0.7 * noise + 0.5 * tone * trem) * env
    sig[-int(SR * 0.02):] *= np.linspace(1, 0, int(SR * 0.02))
    return _norm(sig, 0.7)


def braam(seed: int, length: float = 4.0, root: float = 55.0) -> np.ndarray:
    """Low brassy swell (detuned saws through a closing low-pass)."""
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for mult in (1, 1.5, 2):
        for det in (-0.6, 0, 0.7):
            f = root * mult * 2 ** (det / 100 * 12 / 12)
            ph = 2 * np.pi * f * t + rng.uniform(0, 6)
            saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
            sig += saw / mult
    sig = _fft_filter(sig, 30, 900)
    env = np.minimum(1, t / 0.08) * np.exp(-t * 0.7)
    return _norm(reverb(sig * env, 4.0, 0.4, seed, 2500), 0.8)


def shimmer(seed: int, length: float = 3.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for f in rng.choice([1318.5, 1567.98, 1975.5, 2349.3, 2637.0, 3136.0], 4, replace=False):
        sig += np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(4, 9) * t))
    env = np.minimum(1, t / 0.6) * np.exp(-t * 1.1)
    return _norm(reverb(sig * env, 4.0, 0.6, seed, 9000), 0.5)


def tick(seed: int) -> np.ndarray:
    """Soft digital tick for on-screen text."""
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 60) + 0.5 * np.sin(2 * np.pi * 900 * t) * np.exp(-t * 40)
    return _norm(reverb(sig, 1.0, 0.3, seed, 8000), 0.5)


def drone(length: float, seed: int, root: float = 41.2) -> np.ndarray:
    """Dark evolving bed for cold opens: detuned low sines + breathing filtered noise."""
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for mult, amp in ((1, 1.0), (1.5, 0.35), (2, 0.5), (3, 0.15)):
        for det in (-0.25, 0.25):
            sig += amp * np.sin(2 * np.pi * root * mult * (1 + det / 100) * t + rng.uniform(0, 6))
    air = _fft_filter(rng.standard_normal(n), 150, 900) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 7.0))
    sig = sig * (0.8 + 0.2 * np.sin(2 * np.pi * t / 11.0)) + 1.6 * air
    fade = int(SR * 1.5)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return _norm(sig * env, 0.6)


def heartbeat(seed: int) -> np.ndarray:
    n = int(SR * 1.0)
    t = np.arange(n) / SR
    def thump(t0, a):
        tt = np.clip(t - t0, 0, None)
        return a * np.sin(2 * np.pi * (45 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18) * (t >= t0)
    return _norm(thump(0, 1.0) + thump(0.28, 0.7), 0.8)


def swell(seed: int, length: float = 3.0) -> np.ndarray:
    """Slow rising breath of air and a soft major-ish chord: a gentle approach into a reveal."""
    rng = np.random.default_rng(seed)
    n = int(SR * length)
    t = np.arange(n) / SR
    p = t / length
    air = _sweep_bandpass(rng.standard_normal(n), 400, 2600, 0.7)
    pad = np.zeros(n)
    for k, f in enumerate((110.0, 164.81, 220.0, 277.18, 329.63)):
        pad += np.sin(2 * np.pi * f * (1 + 0.004 * p) * t + rng.uniform(0, 6)) / (1 + 0.6 * k)
    env = (np.sin(np.pi / 2 * p) ** 3) * (1 - np.clip((p - 0.92) / 0.08, 0, 1) * 0.6)
    sig = (0.35 * air + 0.6 * pad) * env
    return _norm(reverb(sig, 5.0, 0.55, seed, 4000), 0.6)


def deep(seed: int) -> np.ndarray:
    """Soft, low bloom: felt more than heard, with a long dark tail (no click, no crack)."""
    n = int(SR * 3.5)
    t = np.arange(n) / SR
    f = 36 + 18 * np.exp(-t * 2.5)
    att = np.clip(t / 0.06, 0, 1)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * att * np.exp(-t * 1.1)
    body = np.sin(2 * np.pi * 73.4 * t) * att * np.exp(-t * 2.2) * 0.35
    sig = _fft_filter(sub + body, 20, 400)
    return _norm(reverb(sig, 5.5, 0.5, seed, 1200), 0.85)


SFX = {"boom": boom, "whoosh": whoosh, "riser": riser, "braam": braam, "shimmer": shimmer, "tick": tick,
       "heartbeat": heartbeat, "swell": swell, "deep": deep}


def effect(name: str, seed: int, **kw) -> np.ndarray:
    return SFX[name](seed, **kw)


# ---- music -------------------------------------------------------------------------------------------------
def music_credits() -> dict:
    p = MUSIC_DIR / "credits.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def credit_line(track: str) -> str:
    c = music_credits().get(track)
    if not c:
        return ""
    return f'"{c["title"]}" Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0'


def _tracks_for(mood: str) -> list[Path]:
    credits = music_credits()
    all_tracks = sorted(MUSIC_DIR.glob("*.mp3"))
    good = [p for p in all_tracks if mood in credits.get(p.name, {}).get("moods", [])]
    return good or all_tracks


def music_section(length: float, mood: str, rng: random.Random, avoid: set[str]) -> tuple[np.ndarray, list[str]]:
    """`length` seconds of stereo music for one mood, tracks chained with 4 s crossfades, loudness matched."""
    tracks = [t for t in _tracks_for(mood) if t.name not in avoid] or _tracks_for(mood)
    rng.shuffle(tracks)
    need = int(length * SR)
    out = np.zeros((0, 2), np.float32)
    names: list[str] = []
    xf = int(4 * SR)
    for tr in tracks[:8]:
        if len(out) >= need:
            break
        total = duration_of(tr)
        if total < 25:
            continue
        start = rng.uniform(0, max(0.0, total - 60)) if total > 120 and not len(out) else 0.0
        sig = load(tr, 2, start, min(total - start, (need - len(out)) / SR + 6))
        sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * 0.1
        names.append(tr.name)
        if not len(out):
            fi = int(SR * 2)
            sig[:fi] *= np.linspace(0, 1, fi)[:, None]
            out = sig
            continue
        ramp = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
        k = min(xf, len(out), len(sig))
        out[-k:] = out[-k:] * (1 - ramp[-k:]) + sig[:k] * ramp[-k:]
        out = np.concatenate([out, sig[k:]])
    if len(out) < need:
        out = np.pad(out, ((0, need - len(out)), (0, 0)))
    return out[:need], names


# ---- mix ----------------------------------------------------------------------------------------------------
def polish_voice(src: Path, dst: Path) -> None:
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-af", VOICE_CHAIN, "-ar", str(SR), "-ac", "1",
                    str(dst)], check=True)


def duck_curve(voice: np.ndarray, base_db: float, duck_db: float) -> np.ndarray:
    hop = SR // 100
    frames = len(voice) // hop + 1
    padded = np.pad(voice, (0, frames * hop - len(voice)))
    rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1))
    active = np.clip((20 * np.log10(rms + 1e-9) + 48) / 14, 0, 1)
    g = np.empty(frames)
    cur = 0.0
    for i, a in enumerate(active):
        coef = 0.25 if a > cur else 0.025      # fast duck, slow musical release (swells between lines)
        cur += (a - cur) * coef
        g[i] = cur
    gains = db(base_db) * db(duck_db) ** g
    return np.interp(np.arange(len(voice)) / hop, np.arange(frames), gains).astype(np.float32)


def mix(work: Path, duration: float, voice_events: list[tuple[float, Path]],
        sfx_events: list[tuple[float, np.ndarray, float, float]],
        music_plan: list[tuple[float, float, str]], cfg: dict, seed: int) -> tuple[Path, list[str]]:
    """voice_events: (t, wav). sfx_events: (t, mono signal, gain dB, pan -1..1). music_plan: (start, end, mood).
    Returns the mastered stereo wav and the music tracks used."""
    n = int(duration * SR) + SR
    voice = np.zeros(n, np.float32)
    for at, wav in voice_events:
        pol = wav.with_name(wav.stem + "_p.wav")
        if not pol.exists():
            polish_voice(wav, pol)
        sig = load(pol)
        i = int(at * SR)
        seg = sig[: max(0, n - i)]
        voice[i:i + len(seg)] += seg
    v_rms = np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2)) if np.any(np.abs(voice) > 0.01) else 0.1
    voice *= 0.16 / (v_rms + 1e-9)

    rng = random.Random(seed)
    music = np.zeros((n, 2), np.float32)
    used: list[str] = []
    mc = cfg.get("music", {})
    for k, (a, b, mood) in enumerate(music_plan):
        if mood == "none" or b - a < 1:
            continue
        sec, names = music_section(b - a + 3.0, mood, rng, set(used))
        used += names
        i0 = int(a * SR)
        seg = sec[: max(0, n - i0)].copy()
        fin = int(SR * (1.5 if k else 0.3))
        seg[:fin] *= np.linspace(0, 1, fin)[:, None]
        tail = int((b - a) * SR)
        fo = int(SR * 3.0)
        if tail < len(seg):
            ln = min(fo, len(seg) - tail)
            seg[tail:tail + ln] *= np.linspace(1, 0, ln)[:, None]
            seg[tail + ln:] = 0
        music[i0:i0 + len(seg)] += seg
    music *= duck_curve(voice, mc.get("level_db", -1.0), mc.get("duck_db", -11.0))[:, None]

    fx = np.zeros((n, 2), np.float32)
    for at, sig, gain, pan in sfx_events:
        i = int(max(0.0, at) * SR)
        seg = sig[: max(0, n - i)] * db(gain)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        fx[i:i + len(seg), 0] += seg * l * 1.41
        fx[i:i + len(seg), 1] += seg * r * 1.41

    total = music + fx + voice[:, None]
    total = total[: int(duration * SR)]
    fade = int(SR * 2.5)
    total[-fade:] *= np.linspace(1, 0, fade)[:, None]
    raw = work / "mix_raw.wav"
    save_stereo(total / max(1.0, float(np.abs(total).max())), raw)
    out = work / "mix.wav"
    loudnorm(raw, out)
    return out, used


def loudnorm(src: Path, dst: Path, target: float = -14.0) -> None:
    first = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src), "-af",
                            f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True, check=True).stderr
    j = json.loads(first[first.rfind("{"): first.rfind("}") + 1])
    af = (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", str(SR), str(dst)], check=True)
