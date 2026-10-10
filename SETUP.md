# One-time setup

Do these once. After step 6 the channel runs by itself: every day GitHub makes the next video, uploads it, sets the
thumbnail and subtitles, and schedules it for a random minute between 4:15 and 5:00 am IST. When fewer than 14
scripts are waiting, Claude researches and writes new ones. If anything ever fails, GitHub emails you.

Until Google approves the API audit (step 4, usually 2–4 weeks), YouTube keeps every API upload **private**, so
the first weeks are scheduled by hand from videos rendered on this PC (step 7, about 10 minutes for a whole week).

---

## 1. YouTube channel (5 minutes)

1. Verify the channel with your phone number: <https://www.youtube.com/verify>
   This unlocks custom thumbnails (essential) and videos longer than 15 minutes.
2. In YouTube Studio → Settings → Channel → Advanced settings, set audience to **No, not made for kids**.

## 2. Google Cloud project (10 minutes, partly done already)

You already created a project and OAuth client (client ID starting `3612369792-`). Check these on it:

1. YouTube Data API v3 is **Enabled**: <https://console.cloud.google.com/apis/library/youtube.googleapis.com>
2. Google Auth Platform → Audience: <https://console.cloud.google.com/auth/audience>
   Publishing status must be **In production** (click *Publish app*). In "Testing" the login expires every 7 days
   and the autopilot would stop. You do not need Google's app verification for your own channel.
3. Clients: <https://console.cloud.google.com/auth/clients> → your **Desktop app** client → *Download JSON*.
   Save it in this folder as `.secrets/client_secret.json` (create the `.secrets` folder; it is never uploaded).
4. Note the **project number** (Cloud console home page, "Project info" card). You need it for step 4.

## 3. Connect the channel (2 minutes)

In this folder, run:

```
.venv\Scripts\python tools\authorize.py .secrets\client_secret.json
```

A browser opens. Pick the account, then **choose the channel** (if it is a brand channel, pick the channel, not your
personal name). If Google says the app is unverified: *Advanced → Go to … (unsafe)* → Allow. This saves
`.secrets/youtube.json` with three values you need in step 6.

## 4. Ask Google to lift the "private only" limit (10 minutes to fill in, 2–4 weeks to approve)

Form: <https://support.google.com/youtube/contact/yt_api_form>, choose the audit / quota extension request.
Suggested answers:

- **API client description**: "A private tool that uploads my own documentary videos to my own YouTube channel
  once a day, schedules their publish time, and sets their thumbnail and captions. It is used by one person (the
  channel owner). It does not access any other user's data and is not offered to the public."
- **API services used**: videos.insert, thumbnails.set, captions.insert, videos.list, playlistItems.list,
  channels.list.
- **Number of users**: 1. **Channels**: 1 (yours).
- If it asks for a link to the client, use your GitHub repository from step 5. If it asks for a privacy policy,
  say the tool stores no data except its own OAuth token, kept as an encrypted GitHub secret.

Google answers by email. Do step 6c only after the approval email arrives.

## 5. GitHub (10 minutes)

1. Create an account if you have none: <https://github.com/signup>
2. Create a new **public** repository (public repositories get free, unlimited and faster daily runs; your secrets
   stay secret): <https://github.com/new>, named for example `cosmos-autopilot`, with no README.
3. Tell Claude the repository address. It pushes this project there (Git opens a browser once to sign you in).
4. Make sure failure emails are on: <https://github.com/settings/notifications> → *Actions* → *Only notify for
   failed workflows* is ticked.

## 6. Secrets (5 minutes)

On GitHub: your repository → **Settings → Secrets and variables → Actions → New repository secret**
(direct link: `https://github.com/<you>/<repo>/settings/secrets/actions/new`).

a. **Claude (writes new scripts when the queue runs low)**
   - Create an API key: <https://console.anthropic.com/settings/keys> (add credits first:
     <https://console.anthropic.com/settings/billing>; about $1 per new episode, roughly $30 a month).
   - Optional safety: set a monthly limit at <https://console.anthropic.com/settings/limits>.
   - Secret name `ANTHROPIC_API_KEY`, value = the key.

b. *(Nothing to add yet for YouTube.)*

c. **After Google's approval email**, add three secrets from `.secrets/youtube.json`:
   `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`.
   From the next morning run on, the workflow takes over. It recognises the videos you scheduled by hand (by
   title), skips the days already taken, and continues with the next episode.

## 7. Until approval: schedule the rendered videos by hand

Videos rendered on this PC are in `out/`, and `out/UPLOAD.md` lists for each one: the file, thumbnail and
subtitles, the exact title, description and tags to paste, and the publish date and time (IST). In YouTube Studio:
*Create → Upload videos*, paste, add the thumbnail and the `.srt` subtitles, *Not made for kids*, then
*Visibility → Schedule* at the listed time. To render more: `.venv\Scripts\python tools\batch.py 008 014`.

## Checking on it

- Every run: your repository → **Actions** tab (green = video scheduled; the log shows the YouTube link).
- Test a run without uploading: Actions → *Daily episode* → *Run workflow* → tick *Test mode*. The finished video
  appears under the run's *Artifacts*.
- Change the time window, voice, or queue size in `config.yaml` (edit it on GitHub; the next run uses it).
