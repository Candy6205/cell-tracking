# Strategy

> Companion to `progress.md` (experiment log), `holdout-split.md`,
> `public-solutions-recon.md`, and `engineering-rules.md`.

*Rewritten 2026-08-20, after the 0.876 submission and the forum recon.*

---

## Where we are

| | public LB | local, 12 held-out `6bba` |
|---|---|---|
| our classical DoG pipeline | 0.681 | ~0.45 |
| official pipeline, greedy linking | — | 0.8847 |
| **official + ILP + app 0.0 / disapp 1.5** | **0.876** | 0.9269 |
| **live public frontier, read from the LB 2026-08-20** | **0.957 #1 / 0.935 #20** | — |

The 0.966 and 0.952 *notebook badges* are **pre-patch numbers** carrying a
division-metric exploit that was closed on 2026-07-17 and rescored on
2026-07-23. Stripped of it they sit around 0.87-0.88. See
`public-solutions-recon.md`.

**But the badges are not the frontier.** Read straight off the leaderboard on
2026-08-20, now that the connector authenticates: **#1 is 0.957, #20 is 0.935**,
and every one of those submissions is post-rescore. **The real gap is +0.059 to
the top 20 and +0.081 to first.** The earlier "frontier ~0.914-0.929" figure was
inferred from badges while the leaderboard endpoint was blocked; it was low by
0.02-0.03.

The leaderboard is also very dense. 0.681 ranked 1940/2470; 0.876 ranks
1560/2560. **+0.195 of score bought 380 places**, so the remaining +0.059 has to
pass ~1540 teams stacked on the public-notebook plateau. Incremental gains will
not show up as rank until that plateau is cleared.

Our submission contains no forks at all, so `summarise()` drops the division
term entirely: **0.876 is our pure adjusted edge Jaccard on the hidden test.**

---

## The three things that actually moved the score

1. **Fixing the local scorer** (2026-08-18). The edge-FP rule was wrong in the
   flattering direction — one matched endpoint is enough, not two. FP went 2 ->
   289 on a 24-dataset sample. Every tuning decision before that rode on a
   metric that does not exist. Now cross-checked against the official scorer:
   edge TP/FP/FN identical, `adj_J` agreeing to five decimals.
2. **Measuring where the loss is.** `error_analysis.py` opened all 10313 GT
   edges: 60% of failures were `DROPPED` (both endpoints detected, no edge
   emitted), not confusion, not detection. That pointed straight at the greedy
   linker's `max_parents_per_node = 1` first-come-first-served rule.
3. **ILP linking.** `--use-ilp --ilp-appearance-weight 0.0
   --ilp-disappearance-weight 1.5`: +0.0422 locally, improved **12 of 12**
   datasets. Most of the gain is the node-count penalty (0.0140 -> 0.0009), not
   the raw Jaccard — the solver prefers long continuous tracks and discards
   scattered nodes.

---

## The 0.051 local-vs-LB offset

Not yet decomposed, but the leading explanation changed:

- **The hidden test is a third embryo.** The host confirmed no overlap in
  `embryo_id` between train and test. It is neither `44b6` nor `6bba`.
- **Our holdout is the easiest slice that exists.** `6bba` has a 7.47%
  duplicate-frame rate affecting 89.1% of its movies; `44b6` has exactly zero.
  A frozen frame is a free link.
- **Density mix cannot explain it alone.** Fitting J against log density on our
  12 and extrapolating: even a test made entirely of `44b6`-like datasets
  predicts 0.885, still above the observed 0.876.

**Operationally: local scores are for paired deltas only.** Never treat the
absolute number as a target.

---

## Plan

*Rewritten 2026-08-20 after the candidate cache and the stranded audit. The
infrastructure is built and validated; what changed is the ranking.*

### What we now know, in three numbers

| | |
|---|---|
| GT edges unreachable at `threshold` 0.5 | **278 of the 450 FN — 62%** |
| stranded targets they hide among | **22,474**, of which ~98.8% are spurious |
| false positives at which recovering all 278 breaks even | **~300** |

The gate really is throwing away most of what remains, and the prize is
narrow: 278 needles in a 22,474 haystack. Full detail in `progress.md`.

### 1. The 2-D replay sweep — running, CPU, no quota

`threshold` {0.50, 0.35, 0.25, 0.15} x `disappearance_weight` {1.0, 1.5,
2.5}, replayed from the cache. The anchor cell reproduces the submitted
0.9269 exactly, so the table is trustworthy.

**Predicted result**, worth writing down before reading it:
`appearance_weight = 0.0` makes starting a track free while
`disappearance_weight = 1.5` penalises ending one, so the solver is biased
towards linking. Once the gate opens onto a pool that is 98.8% junk, that
bias is a liability. **Expect the winner to pair a lower threshold with a
LOWER disappearance weight.** If instead the whole table sits below the
anchor, the haystack won and the lever is dead — say so and move to 2.

### 2. Reduce over-detection — the multiplier, and never tried

The 22,474 spurious nodes are charged twice: the node-count penalty, and
burying the 278 recoverable edges at 1.2%. **Halve the spurious pool and the
needle-in-haystack ratio halves with it**, moving break-even in our favour.
These two lines multiply, they do not add.

`cache_det_sweep.py` tests `det-threshold` at 0.995 and 0.999 against the
submitted 0.99 — one inference pass each, ~11 min. Read the ratio: stranded
falling faster than nodes means the cut lands on the useless detections.

This contradicts `public-solutions-recon.md` item 3, which proposed sweeping
det-threshold **down** to 0.95 by analogy with the public tier. The audit
points up. One of the two is wrong and the run settles it.

Why the project missed this: `detect_ceiling` reads 0.98-0.99 and was taken
to mean detection was solved. **It counts missed cells only.** `N/N_true`
from 1.01 to 1.30 says up to 30% of predicted nodes are invented, and
nothing has ever measured that as a cost.

### 3. CPU only, on predictions we already have

- **Orphan-node forks** (`division_recovery.py`, written but never run). Keep
  the linker 1:1, then add forks in a second pass into parentless nodes only,
  capped per movie. The public tier measures **+0.0129** with the official
  scorer, and it cannot displace a true positive by construction.
- **`MIN_TRACK_LEN = 3` pruning.** Expect close to nothing: ILP already holds
  the node-count penalty at 0.0009. The public tier needs it because its
  linker leaves scattered nodes; ours does not. Measure it, do not budget for
  it.

### 4. The `44b6` blind spot — one inference pass, and it needs care

Every number we own comes from `6bba`. A naive `44b6` holdout **cannot**
answer the offset question: `44b6` runs a median 327 cells/frame against
`6bba`'s 97 and `corr(edge J, density) = -0.68`, so it would score lower for
reasons unrelated to what the weights saw.

`survey_match.py` picks the comparison. Strict density pairing was tried and
discards the low-density half — `44b6` has almost nothing below 65
cells/frame — so the deliverable is **10 datasets spanning the overlap in log
density**, compared as fitted J-vs-density curves. Density becomes a
covariate instead of a confound.

### 5. Only if 1 and 2 both pay

TTA over the linker features, and one-frame gap closing with synthetic
midpoints. Both cost runtime, and there is 2.3x of inference budget spare
(~5.2 h of the 12 h limit), so they are affordable — but they are refinements
of a pipeline whose main leak may still be open.
### Deliberately not doing

- **Global knob tuning.** The 0.952's author swept scalars extensively and
  posted the log: 0.950 / 0.950 / 0.946 / 0.949 / 0.950. Exhausted.
- **Chasing the division term by lowering `--ilp-division-weight`.** Measured
  8:1 against: divJ 0 -> 0.0308 buys +0.0031 while adj_J loses 0.0245. A fork is
  taken iff `edge_prob > division_weight - appearance_weight`, so the knob
  relaxes the bar for every short spurious second edge while real division edges
  — median 5.72 um against 1.82 um for ordinary edges — stay unlikely. Wrong
  direction. The orphan-fork pass is the right one.
- **Elongation as a division feature.** Measured AUC 0.54. Dead.
- **Frame striding to fit the runtime.** Structurally scores 0.0.

---

## What we keep

| asset | still useful? |
|---|---|
| the local scorer, cross-checked against the official one | **yes — this is the differentiator** |
| `error_analysis.py` — per-edge failure attribution | **yes — it found the ILP win** |
| per-dataset survey statistics (density, displacement) | yes — they predict the failure modes |
| offline wheels + submission path, verified end to end | yes |
| `cell_count.py` | yes — the public tier hard-codes counts from the visible 4, which will not survive the swap |
| Hungarian linking, DoG detection | superseded |
| DeepCenter | **no** — the public tier disables it entirely |

---

## Leaderboard resolution

Public LB = 29% of the test data; final standings use the other 71%. A single
public submission resolves roughly +-0.03, so 0.916 vs 0.89 is within noise.
Local comparisons are **paired** — same datasets, one switch flipped — and far
more sensitive than any leaderboard delta.

Final ranking allows selecting 2 submissions. Select by local validation; expect
a shakeup.

---

*Last updated: 2026-08-20*
