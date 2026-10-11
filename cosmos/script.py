"""Episode scripts: parse `episodes/*.md` into sections and paragraphs.

Format (see episodes/README.md for the full guide):

    ---                                   front matter: title, card, thumb, thumb_query, mood, tags, fallback,
    title: ...                            teaser, say (pronunciations "Word=spoken | Word2=spoken")
    ---
    [cold]                                the cold open (hook), before the title sequence
    > v: query | i: query | query         visual searches for the paragraph below (v: video, i: image,
    Paragraph text ... {{shown}} ...      c: Wikimedia Commons, n: NASA, otherwise anything)
    [title]                               the title sequence
    ## Chapter title                      a chapter (optional "mood: wonder" line right after it)
    [outro]                               the closing section; "{next}" becomes the next episode's teaser

Inline marks: {{phrase}} shows the phrase on screen as it is spoken; {{spoken|SHOWN}} speaks one thing and
shows another; [pause] is a dramatic pause.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODES_DIR = ROOT / "episodes"
MOODS = ("mystery", "wonder", "epic", "dread")

PAUSE = "[pause]"


@dataclass
class Emphasis:
    spoken: str          # words as spoken (used to find the moment in the narration)
    shown: str           # text put on screen
    first_word: int      # index of the first spoken word of the phrase within the paragraph's words
    n_words: int


@dataclass
class Part:
    """A piece of narration between pauses."""
    text: str                          # spoken text (marks removed, pronunciations NOT applied)
    emphasis: list[Emphasis] = field(default_factory=list)


@dataclass
class Paragraph:
    queries: list[str]
    parts: list[Part]
    raw: str

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.parts)


@dataclass
class Section:
    kind: str                          # cold | title | chapter | outro
    title: str = ""
    mood: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Episode:
    key: str
    path: Path
    meta: dict
    sections: list[Section]

    @property
    def title(self) -> str:
        return self.meta["title"]

    @property
    def card(self) -> str:
        return self.meta.get("card") or self.meta["title"]

    @property
    def mood(self) -> str:
        return self.meta.get("mood", "mystery")

    @property
    def tags(self) -> list[str]:
        return [t.strip() for t in self.meta.get("tags", "").split(",") if t.strip()]

    @property
    def fallback(self) -> list[str]:
        return _split_bar(self.meta.get("fallback", ""))

    @property
    def thumb_queries(self) -> list[str]:
        return _split_bar(self.meta.get("thumb_query", "")) or self.fallback

    @property
    def pronunciations(self) -> dict[str, str]:
        out = {}
        for item in _split_bar(self.meta.get("say", "")):
            if "=" in item:
                k, v = item.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def chapters(self) -> list[Section]:
        return [s for s in self.sections if s.kind == "chapter"]

    def word_count(self) -> int:
        return sum(len(p.text.split()) for s in self.sections for p in s.paragraphs)


class ScriptError(ValueError):
    pass


def _split_bar(s: str) -> list[str]:
    return [x.strip() for x in s.split("|") if x.strip()]


def words_of(text: str) -> list[str]:
    return [w for w in text.split() if re.sub(r"[^\w]", "", w)]


def _parse_part(raw: str) -> Part:
    emph: list[Emphasis] = []
    out = []
    pos_words = 0
    i = 0
    for m in re.finditer(r"\{\{(.+?)\}\}", raw):
        before = raw[i:m.start()]
        out.append(before)
        pos_words += len(words_of(before))
        inner = m.group(1)
        spoken, shown = (inner.split("|", 1) + [""])[:2] if "|" in inner else (inner, inner)
        spoken, shown = spoken.strip(), (shown.strip() or spoken.strip())
        n = len(words_of(spoken))
        emph.append(Emphasis(spoken=spoken, shown=shown, first_word=pos_words, n_words=max(1, n)))
        out.append(spoken)
        pos_words += n
        i = m.end()
    out.append(raw[i:])
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return Part(text=text, emphasis=emph)


def _parse_paragraph(queries: list[str], raw: str) -> Paragraph:
    pieces = [p.strip() for p in raw.split(PAUSE)]
    parts = [_parse_part(p) for p in pieces if p]
    return Paragraph(queries=queries, parts=parts, raw=raw)


def parse(path: Path) -> Episode:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.match(r"---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ScriptError(f"{path.name}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    if "title" not in meta:
        raise ScriptError(f"{path.name}: no title")
    sections: list[Section] = []
    cur: Section | None = None
    queries: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf, queries
        raw = " ".join(x.strip() for x in buf).strip()
        if raw:
            if cur is None:
                raise ScriptError(f"{path.name}: text before the first section")
            cur.paragraphs.append(_parse_paragraph(queries, raw))
            queries = []
        buf = []

    for line in m.group(2).splitlines():
        s = line.strip()
        if not s:
            flush()
            continue
        if s in ("[cold]", "[title]", "[outro]"):
            flush()
            cur = Section(kind=s[1:-1])
            sections.append(cur)
            continue
        if s.startswith("## "):
            flush()
            cur = Section(kind="chapter", title=s[3:].strip())
            sections.append(cur)
            continue
        if cur is not None and cur.kind == "chapter" and not cur.paragraphs and not buf and \
                re.match(r"mood:\s*\w+$", s):
            cur.mood = s.split(":", 1)[1].strip()
            continue
        if s.startswith(">"):
            flush()
            queries = _split_bar(s[1:])
            continue
        buf.append(s)
    flush()
    ep = Episode(key=path.stem, path=path, meta=meta, sections=sections)
    problems = validate(ep)
    if problems:
        raise ScriptError(f"{path.name}: " + "; ".join(problems))
    return ep


def validate(ep: Episode) -> list[str]:
    p = []
    kinds = [s.kind for s in ep.sections]
    if kinds[:2] != ["cold", "title"]:
        p.append("must start with [cold] then [title]")
    if kinds[-1:] != ["outro"]:
        p.append("must end with [outro]")
    if len(ep.chapters()) < 3:
        p.append("needs at least 3 chapters")
    for s in ep.sections:
        if s.kind != "title" and not s.paragraphs:
            p.append(f"empty section {s.kind} {s.title}")
        if s.mood and s.mood not in MOODS:
            p.append(f"unknown mood {s.mood}")
        for para in s.paragraphs:
            if not para.queries:
                p.append(f"paragraph without visuals: {para.raw[:40]}...")
    if ep.meta.get("mood", "mystery") not in MOODS:
        p.append("unknown episode mood")
    for k in ("thumb", "tags", "fallback", "teaser"):
        if not ep.meta.get(k):
            p.append(f"missing {k}")
    n = ep.word_count()
    if not 1100 <= n <= 1750:
        p.append(f"{n} words (aim for 1150-1550: about 9-12 minutes at the documentary pace)")
    return p


def load_all() -> list[Episode]:
    """Every valid episode, in posting order. A broken script is skipped with a warning, never fatal."""
    out = []
    for p in sorted(EPISODES_DIR.glob("[0-9]*.md")):
        try:
            out.append(parse(p))
        except ScriptError as e:
            import logging
            logging.getLogger("cosmos.script").warning("skipping %s", e)
    return out


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or sorted(str(p) for p in EPISODES_DIR.glob("[0-9]*.md")):
        try:
            e = parse(Path(f))
            print(f"OK  {Path(f).name}: {e.word_count()} words, {len(e.chapters())} chapters, "
                  f"~{e.word_count() / 150:.1f} min")
        except ScriptError as ex:
            print(f"BAD {ex}")
