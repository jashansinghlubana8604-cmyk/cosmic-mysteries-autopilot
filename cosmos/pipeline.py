"""Daily autopilot: keep today's and tomorrow's slots filled with a finished, scheduled episode.

    python -m cosmos.pipeline              # make + upload what is missing
    python -m cosmos.pipeline --dry-run    # make the next episode, do not upload

Safe to run many times a day: an episode is uploaded at most once (state file + hidden tag read back from the
channel), and each publish slot gets at most one video.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos import script, writer  # noqa: E402
from cosmos.youtube import AuthError, QuotaError, YouTube, key_tag  # noqa: E402

STATE = ROOT / "state" / "posted.json"
PENDING_THUMBS = ROOT / "state" / "thumbs"       # thumbnails YouTube refused (channel not verified yet): retried
log = logging.getLogger("cosmos.pipeline")


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"episodes": {}, "failed": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")


def commit_state(msg: str) -> None:
    if os.environ.get("COSMOS_GIT_PUSH") != "1":
        return
    for cmd in (["git", "add", "state", "episodes"], ["git", "commit", "-m", msg], ["git", "pull", "--rebase", "-X", "ours"],
                ["git", "push"]):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode and cmd[1] not in ("commit",):
            log.warning("%s: %s", " ".join(cmd), r.stderr[-300:])


def set_thumbnail(yt, st: dict, key: str, vid: str, image: Path) -> None:
    """Set the thumbnail; if YouTube refuses (custom thumbnails not unlocked yet), keep the image in the repo and
    let every later run retry it."""
    if yt.set_thumbnail(vid, image):
        st["episodes"][key]["thumb"] = True
        return
    PENDING_THUMBS.mkdir(parents=True, exist_ok=True)
    keep = PENDING_THUMBS / f"{key}.jpg"
    if image.resolve() != keep.resolve():
        keep.write_bytes(image.read_bytes())
    st["episodes"][key]["thumb"] = False


def retry_thumbnails(yt, st: dict) -> None:
    for key, e in st["episodes"].items():
        img = PENDING_THUMBS / f"{key}.jpg"
        if e.get("thumb") is False and img.exists() and e.get("video_id"):
            if yt.set_thumbnail(e["video_id"], img):
                e["thumb"] = True
                img.unlink()
                log.info("thumbnail set for %s on retry", key)


def slot_time(sc: dict, day) -> tuple[int, int]:
    """Publish time for a day: a fixed publish_time, or a random minute inside publish_window.

    The random minute is seeded by the date, so every run on any machine picks the same time for the same day.
    """
    if sc.get("publish_window"):
        a, b = (int(h) * 60 + int(m) for h, m in (t.split(":") for t in sc["publish_window"]))
        return divmod(random.Random(f"slot-{day.isoformat()}").randint(a, b), 60)
    return tuple(map(int, sc["publish_time"].split(":")))


def slots(cfg: dict, now: datetime) -> list[datetime]:
    sc = cfg["schedule"]
    tz = ZoneInfo(sc["timezone"])
    local = now.astimezone(tz)
    start = datetime.fromisoformat(sc.get("start_date", "2000-01-01")).date()
    out = []
    for d in range(0, int(sc.get("days_ahead", 1)) + 1):
        day = (local + timedelta(days=d)).date()
        if day < start:
            continue
        hh, mm = slot_time(sc, day)
        t = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
        if t > now + timedelta(minutes=int(sc.get("min_lead_minutes", 45))):
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    st = load_state()
    episodes = script.load_all()
    now = datetime.now(ZoneInfo("UTC"))

    yt = None
    channel = cfg["channel"]["name"]
    if not args.dry_run:
        try:
            yt = YouTube(os.environ.get("YT_CLIENT_ID", ""), os.environ.get("YT_CLIENT_SECRET", ""),
                         os.environ.get("YT_REFRESH_TOKEN", ""))
        except AuthError as e:
            log.error("YouTube login failed: %s", e)
            return 2
        channel = yt.title
        recent = yt.recent()
        tags_seen = {t for v in recent for t in v["tags"]}
        titles_seen = {v["title"].strip().lower() for v in recent}
        # anything already on the channel counts as posted (protects against a lost state commit, and covers
        # videos uploaded by hand in YouTube Studio, which carry the episode's title but not its hidden tag)
        for ep in episodes:
            if ep.key in st["episodes"]:
                continue
            if key_tag(ep.key) in tags_seen or ep.title.strip().lower() in titles_seen:
                v = next(v for v in recent if key_tag(ep.key) in v["tags"]
                         or v["title"].strip().lower() == ep.title.strip().lower())
                st["episodes"][ep.key] = {"video_id": v["id"], "publish_at": v["publishAt"] or v["published"],
                                          "title": v["title"], "recovered": True}
        retry_thumbnails(yt, st)
        save_state(st)
        taken = {(e.get("publish_at") or "")[:10] for e in st["episodes"].values()}
        for v in recent:
            when = v["publishAt"] or v["published"] or ""
            taken.add(when[:10])
    else:
        taken = {(e.get("publish_at") or "")[:10] for e in st["episodes"].values()}

    if not args.dry_run:
        try:
            if writer.ensure(int(cfg.get("writer", {}).get("queue", 14)), log.info):
                commit_state("new episode scripts")
                episodes = script.load_all()
        except Exception as e:  # noqa: BLE001 — a writer failure must never stop today's video
            log.exception("writer failed: %s", e)

    todo = [s for s in slots(cfg, now) if s.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d") not in taken]
    if args.dry_run:
        todo = todo[:1] or [now + timedelta(days=1)]
    remaining = [e for e in episodes if e.key not in st["episodes"] and st["failed"].get(e.key, 0) < 2]
    log.info("channel %s | %d episodes ready to post | slots to fill: %s", channel, len(remaining),
             [s.isoformat() for s in todo])
    if len(remaining) < 10:
        log.warning("ONLY %d UNPOSTED EPISODES LEFT", len(remaining))

    from make import build  # noqa: E402  (heavy imports only when there is work)
    from cosmos.voice import VoiceError  # noqa: E402
    made = 0
    for slot in todo:
        remaining = [e for e in episodes if e.key not in st["episodes"] and st["failed"].get(e.key, 0) < 2]
        if not remaining:
            log.error("no episodes left to post")
            return 3
        ep = remaining[0]
        if ep.meta.get("style") != writer.STYLE and writer.available() and not args.dry_run:
            try:                    # bring an older script up to the current playbook before it goes out
                writer.upgrade(ep.path, log.info)
                ep = script.parse(ep.path)
                commit_state(f"upgraded script {ep.key}")
            except Exception as e:  # noqa: BLE001 — the original script is still good enough to post
                log.exception("upgrade of %s failed, posting it as written: %s", ep.key, e)
        nxt = remaining[1] if len(remaining) > 1 else None
        teaser = nxt.meta["teaser"] if nxt else "another mystery the universe has not explained"
        t0 = time.time()
        try:
            meta = build(ep.path, cfg, ROOT / "out", channel, teaser,
                         log_fn=lambda m: log.info("[%s] %s", ep.key, m))
        except VoiceError as e:     # the TTS service is down, not this episode: keep it, fail the run loudly
            log.error("narration service unavailable, nothing can be made this run: %s", e)
            return 5
        except Exception as e:  # noqa: BLE001 — skip a broken episode, never stop the channel
            log.exception("episode %s failed: %s", ep.key, e)
            st["failed"][ep.key] = st["failed"].get(ep.key, 0) + 1
            save_state(st)
            commit_state(f"episode {ep.key} failed")
            continue
        log.info("made %s in %.0f min (%.1f min long)", ep.key, (time.time() - t0) / 60, meta["duration"] / 60)
        if args.dry_run:
            made += 1
            break
        try:
            vid = yt.upload(Path(meta["video"]), title=meta["title"], description=meta["description"],
                            tags=meta["tags"] + [key_tag(ep.key)], publish_at=slot)
        except QuotaError as e:
            log.error("YouTube quota reached (%s); will retry next run", e)
            return 4
        st["episodes"][ep.key] = {"video_id": vid, "publish_at": slot.astimezone(ZoneInfo("UTC")).isoformat(),
                                  "title": meta["title"], "uploaded": datetime.now(ZoneInfo("UTC")).isoformat()}
        if meta.get("thumbnail"):
            set_thumbnail(yt, st, ep.key, vid, Path(meta["thumbnail"]))
        if meta.get("captions"):
            yt.add_captions(vid, Path(meta["captions"]))
        save_state(st)
        commit_state(f"scheduled {ep.key} for {slot.isoformat()} ({vid})")
        log.info("uploaded %s -> https://youtu.be/%s, publishes %s", ep.key, vid, slot.isoformat())
        made += 1
        for f in (Path(meta["video"]),):
            f.unlink(missing_ok=True)
    log.info("done: %d episode(s) made", made)
    return 6 if todo and not made else 0     # a red run makes GitHub email the owner


if __name__ == "__main__":
    sys.exit(main())
