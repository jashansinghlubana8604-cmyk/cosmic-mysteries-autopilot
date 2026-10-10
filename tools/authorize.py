"""One-time: connect the YouTube channel. Opens a browser; pick the channel and allow.

    python tools/authorize.py path/to/client_secret.json      # or a JSON with client_id / client_secret

Prints YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN and saves them to .secrets/youtube.json
(git-ignored).
"""
from __future__ import annotations

import json
import sys
import functools
print = functools.partial(print, flush=True)
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/youtube.readonly"]


def main() -> None:
    src = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if "installed" not in src:
        src = {"installed": {"client_id": src["client_id"], "client_secret": src["client_secret"],
                             "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                             "token_uri": "https://oauth2.googleapis.com/token",
                             "redirect_uris": ["http://localhost"]}}
    flow = InstalledAppFlow.from_client_config(src, SCOPES)
    creds = flow.run_local_server(port=8765, prompt="consent", access_type="offline", open_browser=True,
                                  authorization_prompt_message="OPEN THIS LINK, choose the channel, then Allow: {url}")
    out = {"client_id": creds.client_id, "client_secret": creds.client_secret, "refresh_token": creds.refresh_token}
    d = ROOT / ".secrets"
    d.mkdir(exist_ok=True)
    (d / "youtube.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("Saved to .secrets/youtube.json")


if __name__ == "__main__":
    main()
