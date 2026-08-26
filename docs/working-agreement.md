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

## Use the scorer's own constants (2026-08-26)

**`MATCH_UM` was 5.0 in every diagnostic in this project. The official
scorer matches at 7.0.** Every recall and miss figure ever reported here
was therefore stricter than the thing being optimised, and by a lot:
6bba's detector recall is 0.9813 at 7.0, not the 0.9561 this log
repeated for two days. Miss rate 1.74%, not 4.45%.

Before measuring anything against ground truth, read the constant out of
`tracking_cellmot/metrics.py` rather than inheriting it from a sibling
script. `evaluate(graph, gt_graph, scale=None, max_distance=7.0)`.

## Say how many datasets an arm actually has (2026-08-26)

`flicker3` reported 44b6 with **49 tracks**. Twelve datasets at 327
cells per frame, annotated at 0.77%, tracks of length ≥3 — forty-nine,
of which thirteen were intermittent.

**Every "three domains agree" verdict in this log has been carrying an
arm of that weight without saying so.** The rule counts domains; it does
not weigh them. Print the n for each arm, every time, so a reader can
see when "three domains" is really two and a rumour.

## An unconditional arm is not a ceiling (2026-08-26)

`gap2` had an arm that dropped the candidate requirement entirely and
bridged **every** gap. It measured +0.0018 / −0.0016 / +0.0013, and I
wrote that down as the upper bound of the gap-closing line: a more
careful selector accepts a subset of those bridges, so it can only land
below.

That is wrong, and the user said so:

> 理想情况下应该是直接插入一个点在一定的情况下可以触发,
> 而不是直接插入一个没有检测出来的点这种条件可以随时触发

**An unconditional arm measures the SUM of its good and bad cases, which
have opposite signs. What it reports is what survives the cancellation,
not the best available.** A gated version sits above it whenever the
gate removes more harm than help — and it can sit above the *original*
too, because it is not a subset of the original's behaviour either.

`gap3` measured it: requiring both joined fragments to be at least 8
nodes long beat the unconditional arm in 6bba (+0.0013 vs +0.0009) and
in 44b6 by 3.6× (+0.0043 vs +0.0012), while both stayed above the
shipped BASE.

**So "drop the condition entirely" answers a different question from
"what is the best condition", and only the first one is cheap.** When
an ungated arm comes back near zero, that is a reason to look for the
cancellation, not a reason to close the line.

## Cross the knobs whose mechanisms touch (2026-08-26)

**Sweeping one thing at a time is not neutral. It is a claim that the
knobs are independent, and that claim needs a reason.**

`gap2` was built to sweep `_gap_close`'s span and candidate requirement
with `SHORT_MIN` fixed at 6. The user stopped it — *"交叉测呢"* — on the
grounds that the two are the same phenomenon from opposite ends: a
dropout longer than one frame shatters a track, the gap repair reaches
only one frame, and the short filter then deletes the fragments. If a
wider bridge repairs the dropouts, there is less for the filter to
delete.

Crossed, that is exactly what happened. **Five gap arms × three domains,
and all fifteen gains shrink by roughly half when `SHORT_MIN` goes
6 → 4:**

| gap arm | 6bba sm6 → sm4 | 44b6 sm6 → sm4 | zh sm6 → sm4 |
|---|---|---|---|
| nocand1 | +0.0018 → **+0.0010** | −0.0016 → **−0.0008** | +0.0013 → **+0.0005** |
| span2i | +0.0019 → +0.0011 | −0.0016 → −0.0008 | +0.0013 → +0.0005 |

**One at a time would have overstated the gap line by about 2x**, and
against a `SHORT_MIN` we had already decided to abandon.

The test to apply before choosing a design: *can I name a mechanism by
which these two knobs act on the same failure?* If yes, cross them. Two
knobs at two values is four arms, not two — usually affordable, and the
interaction is the finding.

Also, from the same run: **the shipped configuration must be a literal
arm, not an assumption.** `gap2`'s rewritten `_gap_close` normalised its
gate by `span + 1`, which at span 1 doubled `GAP1_GATE_UM` from 9 to 18
and quietly moved BASE off the shipped behaviour. The must-reproduce
anchor caught it — adj_J 0.9253 against the recorded 0.9254, div_J
0.1282 against 0.1316 — but only *after* 204 minutes of compute. A
refactor that claims to preserve behaviour should be tested on one
dataset before the sweep is launched.

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
