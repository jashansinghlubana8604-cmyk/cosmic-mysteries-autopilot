# Writing an episode

One file per episode, named `NNN-short-name.md`. Episodes are posted in file-name order, one per day.
Check a script with `python -m cosmos.script episodes/NNN-short-name.md`.

```
---
title: YouTube title (curiosity, under 70 characters)
card: Short on-screen title
thumb: TWO OR THREE | *WORDS*         (| = new line, *word* = yellow accent)
thumb_query: searches for the thumbnail image | another
mood: mystery                        (mystery | wonder | epic | dread)
kicker: Small line above the on-screen title
tags: comma, separated, tags
fallback: generic searches used when a paragraph's own searches run dry
teaser: how the PREVIOUS episode's outro introduces this one ("next, we are going to ...")
say: Word=pronunciation | Other=pronunciation
---

[cold]                cold open: 4-5 paragraphs, start mid-story, end on the question
> v: video search | i: image search | c: Wikimedia search | plain search
Paragraph text. {{spoken words|SHOWN ON SCREEN}} {{shown as spoken}} [pause]

[title]               the warp-speed title sequence

## Chapter Title     5 chapters; optional "mood: wonder" on the next line
...

[outro]               question for the comments + subscribe + {next}
```

Rules that keep quality high:

- 1,300-1,600 words (9-11 minutes). Short sentences. Present the mystery, then the evidence, then the
  explanations, then what is still unknown. Every chapter ends on a reason to keep watching.
- Facts must be checkable. Hedge anything uncertain ("astronomers estimate", "one study suggested").
- 2-4 `{{...}}` on-screen moments per chapter: numbers, names, one-word reveals. A shown text ending
  in `!`, a single upper-case word or anything with a number gets a cinematic hit.
- Searches should name things the archives actually contain: galaxies, nebulae, telescopes, artist's
  impressions, animations, zooms, time-lapses. Avoid people, offices and abstract words like "data".
