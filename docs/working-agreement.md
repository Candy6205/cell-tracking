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

## Three domains, or it does not count (2026-08-25)

**The 12-dataset holdout is 0 for 2 against the leaderboard.** Windowing
measured +0.0019 locally and scored -0.001; the ranker pool measured
**+0.0073**, reproduced bit for bit by two runs with both anchors
holding, and scored -0.001 with hidden-test output within 0.07%. Both
positive local deltas simply were not there.

The boundary is a MAGNITUDE, not a direction. It is tempting to say the
holdout "rejects reliably but selects unreliably" -- **there is no
evidence for that asymmetry** (the user's correction). If it inverts on
a +0.007 it has no claim on a -0.002 either. What it can still do is
reject an effect far outside the ~0.007 inversion scale: `node_filter`'s
-0.03 and `div_gate`'s monotone prune curve are safe rejections.

**So every pipeline evaluation runs in three domains and a change counts
only if its adj_J delta has the SAME SIGN in all of them:**

| domain | why it is there |
|---|---|
| `6bba` (12 holdout) | the recorded anchor lives here |
| `44b6` (12, outside the ranker's pool) | the OTHER imaging batch -- 327 cells/frame against 97, displacement p99 4.90 um against 7.81 |
| Zebrahub (6-8 crops) | a different instrument, and **no public model has seen it** -- which `6bba` cannot claim, since DeepCenter validated on 6bba and the xl detector's split is unknown |

Zebrahub is a TEST domain here, not a training one. Those are different
requirements and conflating them is a mistake already made once: a
Zebrahub-TRAINED node filter lost 0.03 adj_J on competition data
(`biohub-node-filter`), which says nothing about whether a difference
MEASURED there reproduces. Training transfer needs learned parameters to
stay valid across domains; test-domain use needs only the ORDERING to
survive.

Machinery: `post_sweep.py` and `thr3.py` carry all three, including
`zh_load` (from `zh_pipeline.py`) which builds a GT graph from Zebrahub's
own arrays and scores it with the OFFICIAL metric. Reuse them rather
than rebuilding.

**Watch the ranker leak when choosing which model to load.** D187 trained
on 187 of 199, so every `44b6` dataset is in its pool and its edge
probabilities there favour whatever configuration produced its training
candidates. Sweeps that change the DETECTIONS must use D32 and exclude
its 32 datasets from the 44b6 arm. Sweeps that only touch post-link
constants may use either, because the same ranker, detections and ILP
solution appear in both arms and the bias cancels in the delta.

## Say the DENOMINATOR out loud before comparing two rates (2026-08-25)

Seven times in one session I compared numbers whose denominators were
different populations. It is the single most common error this project
has produced, it survives code review because the arithmetic is always
correct, and it has twice reached a written conclusion before the user
caught it.

**The rule: a rate may only be compared to another rate when both
denominators are the same population, or when both are random samples of
it. Write the denominator into the sentence and the bad comparisons
become audible.**

The two live examples, both about the same quantity:

| written as | actually measures |
|---|---|
| "NMS costs 0.91% of ground truth in 6bba" | 0.91% of the **9.71% that were annotated** |
| "NMS costs 13.1% in Zebrahub" | 13.1% of **every cell, all labelled** |

and then, having already recorded that trap, the same entry compared
6bba's 0.0092 to 44b6's 0.0054 — 9.71% annotation coverage against
**0.77%**, a 12x stricter sieve. Both are "sparse", which felt like
enough. It is not: *equally named* is not *equally selected*.

**Sparse ground truth is a hand-picked sample, never a random one.**
Annotators label what they can see, and Zebrahub's recall by brightness
quartile (**0.52 dimmest, 0.74 brightest**) says visibility dominates.
So every sparse-GT rate is optimistic by an unknown amount, and the
amount differs per domain with the coverage.

Three practical consequences:

1. **Within one domain, differences are still clean.** Before-and-after
   on the same annotated set measures that set correctly. Report it that
   way — "0.92% of 6bba's *annotated* cells" — and it stays true.
2. **Across domains, only dense GT can be compared.** Zebrahub is the
   only arm whose denominator is the true cell population. In any
   experiment about what fraction of cells something loses, Zebrahub is
   not one of three equal arms; it is the measurement, and the sparse
   arms are context.
3. This is a different rule from *three domains, or it does not count*
   above, and neither implies the other. That rule is about whether an
   effect generalises. This one is about whether two numbers were ever
   the same kind of number.
