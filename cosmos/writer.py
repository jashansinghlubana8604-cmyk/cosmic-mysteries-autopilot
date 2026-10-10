"""The writer: keeps the queue of episode scripts full, and brings older scripts up to the current standard.

    python -m cosmos.writer --ensure 14      # write new episodes until 14 are waiting to be posted
    python -m cosmos.writer --new            # write one new episode now
    python -m cosmos.writer --upgrade 008    # rewrite an existing script to the playbook (marks it `style: 2`)

Claude researches each topic with web search (recent discoveries first, otherwise a proven evergreen topic the
channel has not covered), writes the script in the episodes/ format following research/playbook.md, and fixes it
until `cosmos.script` accepts it. Needs ANTHROPIC_API_KEY; without it the channel keeps posting the scripts it has.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos import script  # noqa: E402

log = logging.getLogger("cosmos.writer")

MODEL = "claude-opus-5-5"
STYLE = "2"                     # bump when the playbook changes enough that old scripts should be rewritten
STATE = ROOT / "state" / "posted.json"
EXAMPLE = "001-voyager-edge.md"   # the reference script shown to the writer (already at the current standard)
MIN_WORDS, MAX_WORDS = 1350, 1700


class NoKey(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _posted() -> set[str]:
    if not STATE.exists():
        return set()
    return set(json.loads(STATE.read_text(encoding="utf-8")).get("episodes", {}))


def pending() -> list[Path]:
    done = _posted()
    return [p for p in sorted(script.EPISODES_DIR.glob("[0-9]*.md")) if p.stem not in done]


def _system() -> list[dict]:
    """Stable instructions + playbook + format guide + one finished example (cached across calls)."""
    playbook = (ROOT / "research" / "playbook.md").read_text(encoding="utf-8")
    guide = (script.EPISODES_DIR / "README.md").read_text(encoding="utf-8")
    example = (script.EPISODES_DIR / EXAMPLE).read_text(encoding="utf-8")
    report = (ROOT / "research" / "report.md").read_text(encoding="utf-8")
    text = f"""You write narration scripts for a faceless YouTube channel of long-form space-mystery documentaries
(10-13 minutes, one per day, English, global audience). A pipeline turns each script into a finished video
automatically: a neural voice reads the narration, real footage from NASA/ESA/ESO/NOIRLab archives is found from the
`>` search lines, the `{{{{...}}}}` marks become on-screen text, and the front matter becomes the YouTube title,
thumbnail and tags. Nobody edits your script before it is published, so it must be finished, factually correct and
in the exact file format.

Your goal is the one the channel lives by: videos people click, watch to the end and share, built on what has
already been proven to work in this niche, and never on invented claims. Accuracy is the brand: verify every
number, date and name with web search before you use it, hedge what is uncertain, and never present speculation as
discovery. Titles and thumbnails must be things the video truly delivers.

<playbook>
{playbook}
</playbook>

<format_guide>
{guide}
</format_guide>

<outlier_research>
{report}
</outlier_research>

<example_script file="{EXAMPLE}">
{example}
</example_script>

Hard requirements for every script you output:
- {MIN_WORDS}-{MAX_WORDS} spoken words; [cold] then [title] then exactly 5 chapters (`## Title`, optional
  `mood:` line) then [outro]; every paragraph preceded by a `>` line of 2-3 archive searches (v: video, i: image,
  n: NASA, c: Wikimedia Commons) naming things those archives actually contain.
- Front matter keys: title, card, thumb, thumb_style (cinematic | headline | impact), optional thumb_mark: circle,
  thumb_query, mood, kicker, tags, fallback, teaser, say, and `style: {STYLE}`.
- The outro ends with: "And subscribe, because next, we are going to {{next}}." (literally `{{next}}`).
- `say:` gives phonetic spellings for every unusual name the voice might mispronounce.
- Write numbers the way they should be spoken in the narration ("sixty six billion"); use `{{{{spoken|SHOWN}}}}`
  when the screen should show digits.
- Output the complete file inside <episode file="NNN-short-name.md"> ... </episode> and nothing after it."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _client():
    if not available():
        raise NoKey("ANTHROPIC_API_KEY is not set")
    import anthropic
    return anthropic.Anthropic(max_retries=4, timeout=900)


def _converse(messages: list[dict], log_fn=print) -> str:
    """Run one turn to completion (web searches, pause_turn resumes) and return the text Claude wrote."""
    client = _client()
    for _ in range(8):
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=_system(),
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 25}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        u = msg.usage
        log_fn(f"writer: {msg.stop_reason}, in {u.input_tokens} (+{u.cache_read_input_tokens or 0} cached) "
               f"out {u.output_tokens}")
        messages.append({"role": "assistant", "content": [b.to_dict() for b in msg.content]})
        if msg.stop_reason == "pause_turn":
            continue
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"writer declined: {msg.stop_details}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("writer ran out of output tokens")
        return "".join(b.text for b in msg.content if b.type == "text")
    raise RuntimeError("writer: turn still paused after 8 continuations")


def _extract(text: str) -> tuple[str, str]:
    m = re.search(r'<episode file="([^"]+)">\s*(.*?)\s*</episode>', text, re.S)
    if not m:
        raise ValueError("no <episode file=...> block in the reply")
    return m.group(1).strip(), m.group(2).strip() + "\n"


def _check(path: Path) -> list[str]:
    try:
        ep = script.parse(path)
    except script.ScriptError as e:
        return [str(e)]
    problems = []
    n = ep.word_count()
    if not MIN_WORDS <= n <= MAX_WORDS:
        problems.append(f"{n} spoken words; it must be {MIN_WORDS}-{MAX_WORDS}")
    if len(ep.chapters()) != 5:
        problems.append(f"{len(ep.chapters())} chapters; it must be exactly 5")
    if "{next}" not in "".join(p.raw for s in ep.sections if s.kind == "outro" for p in s.paragraphs):
        problems.append("the outro must end with the {next} teaser line")
    if len(ep.title) > 70:
        problems.append(f"title is {len(ep.title)} characters; keep it under 70 (ideally under 60)")
    if ep.meta.get("style") != STYLE:
        problems.append(f"front matter must include `style: {STYLE}`")
    return problems


def _write_until_valid(messages: list[dict], target: Path | None, log_fn=print) -> Path:
    """Ask, validate, and send the problems back until the script passes (3 rounds max)."""
    for attempt in range(3):
        reply = _converse(messages, log_fn)
        name, body = _extract(reply)
        path = target or script.EPISODES_DIR / name
        tmp = path.with_suffix(".draft")
        tmp.write_text(body, encoding="utf-8")
        problems = _check(tmp)
        if not problems:
            tmp.replace(path)
            log_fn(f"writer: wrote {path.name} ({script.parse(path).word_count()} words)")
            return path
        tmp.unlink(missing_ok=True)
        log_fn(f"writer: attempt {attempt + 1} rejected: {'; '.join(problems)}")
        messages.append({"role": "user", "content": "The pipeline rejected that script:\n- " + "\n- ".join(problems)
                         + "\nFix these and output the complete corrected file in the same <episode> format."})
    raise RuntimeError("writer could not produce a valid script in 3 attempts")


def _catalogue() -> str:
    rows = []
    for p in sorted(script.EPISODES_DIR.glob("[0-9]*.md")):
        try:
            ep = script.parse(p)
            rows.append(f"- {p.stem}: {ep.title} (thumb: {ep.meta.get('thumb', '')})")
        except script.ScriptError:
            rows.append(f"- {p.stem}")
    return "\n".join(rows)


def new_episode(log_fn=print) -> Path:
    nums = [int(p.stem[:3]) for p in script.EPISODES_DIR.glob("[0-9][0-9][0-9]-*.md")]
    nxt = f"{(max(nums) + 1) if nums else 1:03d}"
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    prompt = f"""Today is {today}. Write episode {nxt}.

Episodes the channel already has (never repeat a topic or a title angle):
{_catalogue()}

Choose the topic first:
1. Search the web for space and astronomy news from the last 30 days. If there is a real, well-sourced discovery
   or event that fits one of the proven frames in the playbook (something strange / impossible / terrifying /
   record-breaking about a famous object, Voyager, James Webb, a signal, an interstellar object, a nearby world),
   and it has enough solid material and archive footage for 11 minutes, use it.
2. Otherwise pick the strongest evergreen topic from the playbook's proven list and the outlier research that this
   channel has not covered yet.

Then research it properly (at least 6 searches: the key facts, numbers, dates, people, the latest findings, and
any common misconceptions to avoid), and write the script. Name the file {nxt}-short-name.md."""
    return _write_until_valid([{"role": "user", "content": prompt}], None, log_fn)


def upgrade(path: Path, log_fn=print) -> Path:
    """Rewrite an older script to the current standard, keeping its topic, file name and verified facts."""
    old = path.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    prompt = f"""Today is {today}. Here is an older script from this channel, written before the playbook existed:

<old_script file="{path.name}">
{old}
</old_script>

Rewrite it to the standard of the example script and the playbook, keeping the topic and the facts that are
correct. Specifically: a sharper title that follows the proven patterns; a thumb line and style that would win the
click; a cold open that hooks in the first sentence and opens a loop; escalation chapter by chapter; twist beats;
"you" inside the scene; {MIN_WORDS}-{MAX_WORDS} words. Search the web to check every fact and to add anything
newer than the old script (missions, discoveries, revised numbers up to today). Keep the `teaser:` meaning the
same (it introduces this topic) and keep good `>` search lines. Output it as <episode file="{path.name}">."""
    return _write_until_valid([{"role": "user", "content": prompt}], path, log_fn)


def ensure(n: int, log_fn=print) -> int:
    """Write new episodes until at least n are waiting to be posted. Returns how many were written."""
    if not available():
        log_fn("writer: ANTHROPIC_API_KEY not set; not writing new episodes")
        return 0
    wrote = 0
    while len(pending()) < n and wrote < 3:            # at most 3 per run keeps cost and run time bounded
        new_episode(log_fn)
        wrote += 1
    return wrote


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensure", type=int)
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--upgrade")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    say = lambda m: print(m, flush=True)  # noqa: E731
    if args.upgrade:
        hits = [p for p in sorted(script.EPISODES_DIR.glob("[0-9]*.md")) if p.stem.startswith(args.upgrade)]
        if not hits:
            raise SystemExit(f"no episode matches {args.upgrade!r}")
        upgrade(hits[0], say)
    if args.new:
        new_episode(say)
    if args.ensure:
        ensure(args.ensure, say)


if __name__ == "__main__":
    main()
