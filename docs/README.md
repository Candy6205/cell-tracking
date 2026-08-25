# docs/ — the project documents

**These files are the master copy.** Every session edits them here, in the
repository. The Claude Project of the same name holds an older transcription
and is no longer authoritative — if the two disagree, this folder wins.

**The reader is an AI session, not a human.** Write accordingly: newest
first, dated `##` grep anchors, one fact in one place with pointers
elsewhere, superseded claims marked in place (`SUPERSEDED <date>:`), no
narrative that requires reading older entries first.

| file | what it is |
|---|---|
| `working-agreement.md` | **read first**: how to talk to the user, the assembly pattern, and the rules that cost a run each |
| `progress.md` | the log: events and measurements, strictly newest-first; **current state and to-do at the top** |
| `reference.md` | stable lookup facts: metric, train-set survey, Zebrahub, submission mechanics, code map, traps |
| `strategy.md` | direction: what to do next and what we decided NOT to do |
| `kaggle-toolchain.md` | how to actually run things: CLI pushes, connector reads, and their traps |
| `competition-brief.md` | task, data format, the exact metric rules as published |
| `engineering-rules.md` | 11 conventions every script follows, each from something that broke |
| `public-solutions-recon.md` | what the public leaderboard actually contains, read from the forum |
| `holdout-split.md` | which datasets the public weights were trained on — mostly superseded |

Cold start: `working-agreement.md` first, then the top of `progress.md`
(current state + to-do), then `strategy.md`, then grep `reference.md` as
needed. `kaggle-toolchain.md` before running anything on Kaggle.

**This repository is now self-contained.** The working agreement used to
live in a local assistant memory directory that does not travel; it was
migrated here on 2026-08-25 so a session cloning this remote — including
a cloud session on another machine — starts with the same context a local
one has.
