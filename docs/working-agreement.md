# Working agreement — how to run a session on this project

*Migrated 2026-08-25 out of the local assistant memory directory, which
does not travel. Everything a fresh session needs now lives in the repo,
so a cloud session that clones this remote starts with full context.*

---

## How to talk to the user

**Plain language, result first.** Their words: "说的很厉害的样子但是我听不懂"
— it sounds impressive and I can't follow it. And sharper, later: long
reasoning chains do not merely bore them, **the chain drifts somewhere
wrong and the length hides it.**

- Lead with the result and what it changes, in two or three sentences.
- Explain a mechanism only when it changes a decision.
- Never describe a measurement before running it.
- They ask for detail when they want it.
- **No metaphors.** Said directly on 2026-08-25: "求你不要说话用比喻了".
  Say the literal thing — "the detector's recall is 0.65", not "the ruler
  is wrong"; "brightness weights dim voxels less", not "it strips the
  fringe". This governs what is written TO the user; entries in `docs/`
  already use figurative shorthand and are not being rewritten.
- **No unexplained abbreviations.** 2026-08-25: "TTA是什么，神奇的缩写增加了".
  Say the thing — "flip-and-average" / 翻转平均, not "TTA" — or expand it
  the first time. Project-local names coined WITH the user (adj_J, div_J,
  the orphan patch, the repair pool) are fine; field jargon imported from
  outside is not, and switching to an abbreviation for something already
  being discussed in plain words is the worst version of it.

**They will interrogate the provenance of any claim, and they are right
to.** Recalled "facts" drift — quantifiers inflate, inferences harden into
facts. So:

- look numbers up in `reference.md` / `progress.md` **at the moment of
  use**, never answer from recall;
- label anything unverified as unverified;
- keep facts and inferences visibly separate.

This has repeatedly paid: on 2026-08-24 they caught "the same person wrote
both" (it was a transplant from a public solution), "morphology change is
an edge signal" (unresolved confound, which then had to be tested), and
"aren't you changing too much at once" (which turned into the step-1 gate
that caught a real 40% intensity discrepancy).

## These docs are the master copy

`docs/` in this repository is authoritative. A Claude Project of the same
name holds an older transcription and is **not**. Several sessions have
edited this folder directly.

Cold start: the top of `progress.md` (current state and to-do), then
`strategy.md`, then `kaggle-toolchain.md` before running anything on
Kaggle. Grep `reference.md` as needed.

**The reader is an AI session, not a human.** Newest first, dated `##`
grep anchors, one fact in one place with pointers elsewhere, superseded
claims marked in place with `SUPERSEDED <date>:`, no narrative that
requires reading older entries first.

## Running things

Full detail in `kaggle-toolchain.md`. The three things a session gets
wrong if it does not read that file first:

1. **Push with `py kag.py`, never `py -m kaggle`** — the OAuth token has a
   30-minute dead window the SDK does not handle (trap 1b).
2. **Anything starting from cached detections belongs on a CPU kernel.**
   `enable_gpu: false` costs no quota and does not queue; the ILP and the
   official scorer reproduce bit for bit (section 9).
3. **Start one background waiter per run**, and grep it for
   `complete|error|cancel` — silence is indistinguishable from
   still-running (trap 6).

### The assembly pattern

Most experiment scripts here are ASSEMBLED, not written from scratch: a
proven script's preamble and plumbing are cut out by LINE CONTENT and a
new feature block and measurement block are appended. This keeps the
environment, the caches, the holdout and the labels byte-identical to
whatever produced the number being compared against, so an A/B differs
only in what is intended. The assemblers live in the session scratchpad;
the assembled scripts are what gets committed.

Two rules learned the hard way:

- **Locate cuts by line content, never by a search string containing an
  escape** — heredocs through the tool layer eat one level of escaping
  (toolchain trap 8, hit four times on 2026-08-24 alone). Use the file
  editing tools for any patch containing `\n` or `\\`.
- **Run a static undefined-name pass before every push, and do not push
  when it is dirty.** It caught `root`, `GATE_XY`/`GATE_Z` and `POOL`
  missing from assemblies on 2026-08-24, each of which would have died
  at startup and wasted the queue wait. It cannot catch a missing MODULE
  or a data source the inherited preamble asserts on — check those by
  reading the preamble's imports and asserts.

## Always anchor

Every A/B must contain an arm that reproduces a recorded number. When
`poolsize`'s D32 arm returned edge AUC 0.9884 — the exact figure
`ranker_fine` recorded — and adj_J within 0.0002 of `div-gate`, the
+0.0073 delta was attributable to the one thing that varied. When
`morph_probe`'s step 1 did NOT reproduce, the shape of the mismatch
(geometry exact, all four intensity columns off) named the cause in one
step.

**An offline AUC is not a score.** On 2026-08-24 edge AUC anti-predicted
the pipeline three times running. Measure adj_J.
