"""YouTube Data API: scheduled upload, custom thumbnail, closed captions, and a read-back of recent uploads so
the same episode is never posted twice."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httplib2
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger("cosmos.youtube")
RETRY = {500, 502, 503, 504}


class QuotaError(Exception):
    pass


class AuthError(Exception):
    pass


def key_tag(key: str) -> str:
    """Hidden tag tying an upload to its episode file (stable across runs)."""
    return "cm" + hashlib.md5(key.encode()).hexdigest()[:10]


def _reason(e: HttpError) -> str:
    try:
        return json.loads(e.content)["error"]["errors"][0]["reason"]
    except Exception:  # noqa: BLE001
        return ""


class YouTube:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str) -> None:
        if not (client_id and client_secret and refresh_token):
            raise AuthError("YouTube credentials missing (YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN)")
        creds = Credentials(None, refresh_token=refresh_token, client_id=client_id, client_secret=client_secret,
                            token_uri="https://oauth2.googleapis.com/token")
        self.api = build("youtube", "v3", credentials=creds, cache_discovery=False)
        try:
            self.channel = self.api.channels().list(part="snippet,contentDetails", mine=True).execute()["items"][0]
        except RefreshError as e:
            raise AuthError(f"refresh token rejected: {e}") from e

    @property
    def title(self) -> str:
        return self.channel["snippet"]["title"]

    def recent(self, pages: int = 3) -> list[dict]:
        """Recent uploads (including scheduled ones): id, title, tags, publishAt, privacy."""
        pl = self.channel["contentDetails"]["relatedPlaylists"]["uploads"]
        out, token = [], None
        for _ in range(pages):
            r = self.api.playlistItems().list(part="contentDetails", playlistId=pl, maxResults=50,
                                              pageToken=token).execute()
            ids = [i["contentDetails"]["videoId"] for i in r.get("items", [])]
            if ids:
                for v in self.api.videos().list(part="snippet,status", id=",".join(ids)).execute().get("items", []):
                    out.append({"id": v["id"], "title": v["snippet"]["title"], "tags": v["snippet"].get("tags", []),
                                "publishAt": v["status"].get("publishAt"),
                                "published": v["snippet"].get("publishedAt"),
                                "privacy": v["status"]["privacyStatus"]})
            token = r.get("nextPageToken")
            if not token:
                break
        return out

    def upload(self, path: Path, *, title: str, description: str, tags: list[str], publish_at: datetime | None,
               category: str = "28") -> str:
        status = {"privacyStatus": "private" if publish_at else "public", "selfDeclaredMadeForKids": False,
                  "embeddable": True, "containsSyntheticMedia": False}
        if publish_at:
            status["publishAt"] = publish_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        body = {"snippet": {"title": title[:100], "description": description[:4900], "tags": tags,
                            "categoryId": category, "defaultLanguage": "en", "defaultAudioLanguage": "en"},
                "status": status}
        media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=16 * 1024 * 1024, resumable=True)
        req = self.api.videos().insert(part="snippet,status", body=body, media_body=media)
        resp, attempt = None, 0
        while resp is None:
            try:
                _, resp = req.next_chunk()
            except HttpError as e:
                r = _reason(e)
                if r in ("quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
                    raise QuotaError(r) from e
                if e.resp.status not in RETRY or attempt >= 8:
                    raise
                attempt += 1
                time.sleep(min(64, 2 ** attempt) + random.random())
            except (httplib2.HttpLib2Error, ConnectionError, TimeoutError, OSError) as e:
                if attempt >= 8:
                    raise
                attempt += 1
                log.warning("upload hiccup (%s), retrying", e)
                time.sleep(min(64, 2 ** attempt) + random.random())
        return resp["id"]

    def set_thumbnail(self, video_id: str, image: Path) -> bool:
        try:
            self.api.thumbnails().set(videoId=video_id,
                                      media_body=MediaFileUpload(str(image), mimetype="image/jpeg")).execute()
            return True
        except HttpError as e:
            log.warning("thumbnail not set (%s): verify the channel at youtube.com/verify", _reason(e))
            return False

    def add_captions(self, video_id: str, srt: Path) -> bool:
        try:
            self.api.captions().insert(
                part="snippet",
                body={"snippet": {"videoId": video_id, "language": "en", "name": "English", "isDraft": False}},
                media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream")).execute()
            return True
        except HttpError as e:
            log.warning("captions not added (%s)", _reason(e))
            return False
