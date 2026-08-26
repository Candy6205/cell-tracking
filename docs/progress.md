# Progress log — Biohub Cell Tracking During Development

> Reader: a future AI session. Strictly reverse-chronological; the newest
> entry and the state block below are always at the top. Stable lookup facts
> (metric, survey, mechanics, code map, traps) live in `reference.md` — this
> file records events and measurements only. When a later entry overturns an
> earlier claim, the earlier text carries a `SUPERSEDED` mark in place.
> Engineering conventions: `engineering-rules.md` (rule 1: degradation is
> never silent).

## HANDOFF (2026-08-26, replacing the 2026-08-24 one below)

**Best confirmed LB is still 0.921, ref 55707206** (D32 ranker).
**Ref 55772253 is PENDING** — `SHORT_MIN` 6 → 4, one non-comment line
different from ref 55757064, so its control is **0.920**, not 0.921.

### The single most important open question

**Does a three-domain-agreeing result appear on the leaderboard?**

The 12-dataset holdout is 0 for 2 against the LB (windowing +0.0019 →
−0.001; ranker pool +0.0073 → −0.001), and both failures were `6bba`
alone. The pre-registered rule — same sign in `6bba`, `44b6` and
Zebrahub — was built to fix that, and `SHORT_MIN` 6 → 4 is the first
candidate it has ever produced (+0.0025 / +0.0054 / +0.0045). Ref
55772253 tests the RULE, not the constant. Either answer is worth more
than the constant:

- **it appears** → three-domain agreement buys what a single-domain
  holdout does not, and the rule earns its cost;
- **it does not** → the rule is not sufficient either, and no local
  measurement at this scale should drive a submission again.

### What 2026-08-25/26 closed, and what it opened

| line | verdict |
|---|---|
| post-link constants (14 changes) | **one survivor**: `SHORT_MIN` 6 → 4 |
| `SHORT_MIN` curve, 1→7 | a curve, not a spike; deleting length-4 and length-5 tracks loses in **all three** domains |
| NMS radius / density | **closed.** 2 of 3 domains, none significant |
| detector depth | **closed.** signs disagree, nothing significant |
| gap closing (span, interpolation) | **closed.** all arms fail the rule; the span never mattered |
| detection gate 0.15 → 0.12 | **contested.** 6bba +0.0021 and Zebrahub +0.0080 against 44b6 −0.0006. Discarded by the rule; the user's call |
| morphology | closed 2026-08-25 (`MASS_ratio` median 0.994) |

**The one thing OPENED and not yet acted on**: `flicker.py` found that
**no annotated cell is permanently invisible** — 0.0% of 379 GT tracks
are never detected, 28.8% are intermittent, and **48% of dropouts are
longer than the one frame `_gap_close` can bridge**. `gap2` then showed
the obvious repairs do not work, because at 79.8% of missed frames the
heatmap response *vanishes* rather than dipping below threshold. So the
losses are real, identified, and currently unaddressed by anything that
has been tried.

### Read these first

1. **`biohub-div-train` v3 ANSWERED (2026-08-24): yes.** Zebrahub only
   0.8083 on held-out 6bba, 44b6 only (18 pos) 0.4295. The external-data
   line is open. Details and the caveat about the mixture sweep are in
   open item 1 below.
2. **Windowing is REVERTED. Use the global solve.** Three LB points:
   global 0.921, windowed 0.920, windowed-with-seam-constraint 0.920.
   The runtime worry that motivated windowing never materialised -- the
   global solve COMPLETED on the hidden test. Do not re-introduce it
   without evidence that the 12 h cap actually binds.
3. **`total_bytes` in the submission metadata is the only measurement of
   behaviour on the HIDDEN test.** Check it on every submission. It is
   what revealed that the windowed run emitted 10% more output there
   while emitting 2% LESS on the visible 4.

### The three big facts this session established

- **Divisions cannot be learned from competition data.** 151 GT
  divisions in the entire 199-dataset train split, from only 2 embryos
  (26 in 44b6, 125 in 6bba). **124 of them land on detections.**
- **The linker is finished.** The solver already takes every division
  the candidate set contains; widening the gate (isotropic and
  z-anisotropic, both properly swept) only moves divisions from
  "outside gate" to "below floor" and never raises REACHABLE past 5.
- **Zebrahub is another embryo domain, not another problem.** Every
  geometric gap between it and the competition is the size of a
  gap between two embryos within either dataset.

### Assets now in place

| what | where |
|---|---|
| xl detections, ALL 199 train datasets | `biohub-cache-train` (65 MB, 65 s/dataset) |
| xl detections, 41 Zebrahub crops + labels | `biohub-zh-detect` (26 MB) |
| 468,617 validated Zebrahub divisions | `biohub-zh-divisions` (counts only) |
| trained edge ranker | `biohub-ranker-fine` -> `edge_ranker.pkl` |

### Open, in rough order of expected value

1. **PARKED 2026-08-24, user's call. Every CHEAP route is spent; the
   remaining ones all need pixels.** Do not re-run the light experiments.

   What was settled, in order: div-train (Zebrahub alone carries it,
   held-out 6bba AUC 0.808 vs 0.430 for 44b6's 18 positives, below
   chance); div-kinematics (the DAUGHTERS' future is real signal --
   `sep_slope` 1.155 vs 0.082 um/frame, a 14x gap -- while the mother's
   pre-division slow-down does NOT exist in this data, `decel` 0.905 vs
   0.934); div-gate (cannot audit already-selected forks; pruning is
   monotonically worse than doing nothing); div-forks (cannot select
   them either -- rerank buys 1 of 23 removable FPs, generate is worse).

   **Not "unreachable" -- "the model is not good enough".** ~+0.027 sits
   in the rerank direction and the model captured 1/23 of it.

   Why the rest is expensive: **every feature ever built for this reads
   `detcache_*.npz`, i.e. coordinates and detection scores. Nothing reads
   pixels.** So "the mother got smaller / rounder / brighter just before
   dividing" -- the obvious remaining signal, and the one mitotic
   rounding actually produces -- is not expressible at all. Getting it
   means re-reading the raw zarr volumes per candidate, the `det_ablate`
   cost profile (minutes per dataset), not a cheap CPU pass over caches.
   `division_features.py` exists for exactly this question and **has no
   recorded result anywhere** -- written, never run, or run unlogged.

   The other untested lever is data volume: `biohub-zh-divisions`
   validated **468,617** divisions; training uses **2001**, because only
   41 crops were extracted. `PER_CROP = 400` is not binding, the crop
   COUNT is. That one IS cheap -- `zebrahub_extract.py` already exists --
   but nobody has measured whether AP scales with it.
2. **Explain the 10% output gap** between global and windowed on the
   hidden test. The seam hypothesis is DISPROVEN (below). One untested
   idea: windowing charges `disappearance_weight` once per window per
   track instead of once per track, biasing toward over-linking.
3. `biohub-zh-pipeline` died of OOM after 10 of 41 crops and its
   `div_J` came out nan. Its 9 completed crops already answered the
   transfer question; finishing it is optional.
4. **CLOSED, negative**: scale-invariant features for the edge ranker do
   nothing (entry below). The -0.73 density penalty is NOT in the
   ranker's feature space -- per-dataset AUC is flat against density
   while adj_J is not.

## 2026-08-26 -- ⭐⭐⭐⭐⭐ THE NODE-COUNT PENALTY IS THE STORY. Local is UNDER target, so it rewards what the leaderboard punishes.

### The two leaderboard results, and the number that explains both

| submission | change | local, three domains | output | **LB** |
|---|---|---|---|---|
| 55772253 | `SHORT_MIN` 6→4 | +0.0025 / +0.0054 / +0.0045 | +1.61% | **0.920** (0.000) |
| 55780773 | + gate 0.15→0.12 | **+0.0055 / +0.0045 / +0.0121** | +10.3% | **0.918** (−0.002) |

The second was submitted with a **pre-registered prediction of 0.919**,
derived from the output growth alone. It landed at 0.918 — direction
right, magnitude slightly understated.

**The three-domain rule is now 0 for 2**, exactly like the single-domain
holdout it was built to replace. And the second result is the strongest
local number this project has ever produced.

### Why: `N_pred / N_true` on the holdout is 0.977, so the penalty is a BONUS

`J_adj = max(0, J · (1 − 0.1 · (N_pred − N_true)/N_true))`, and on the
12-dataset holdout **11 of 12 datasets sit UNDER the target node count**:

| dataset | true/fr | N_true | N_pred | ratio | penalty factor |
|---|---|---|---|---|---|
| 7b5d3b2c | 58 | 5,780 | 4,990 | **0.863** | **1.0137** |
| 3a1849c2 | 70 | 7,003 | 6,223 | 0.889 | 1.0111 |
| 2312ac41 | 120 | 11,950 | 11,523 | 0.964 | 1.0036 |
| 283bf9f1 | 209 | 20,861 | 20,811 | 0.998 | 1.0002 |
| **3abfe10a** | 521 | 52,065 | 53,504 | **1.028** | 0.9972 |
| **TOTAL** | | 164,698 | 160,857 | **0.977** | **1.0023** |

**Locally, adding nodes helps TWICE** — more correct edges, and a move
toward the target that makes the adjustment term better. On the hidden
test, if `N_pred` is already at or over target, adding nodes helps once
and hurts once. That is a mechanism, and it needs no assumption about
density.

`corr(true cells/frame, N_pred/N_true) = +0.7498`, n=12, p≈0.005 — the
DENSE dataset is the one over target, not the sparse ones.

### ⚠ The user's correction: this is a WARNING, not a conclusion

Two limits, both theirs:

1. **The public split's composition is unknown and need not be
   adversarial to be skewed.** Taking the first or last N datasets is
   enough to give the public set a narrow density band while the hidden
   set spans wider. So LB 0.918 says "adding nodes loses on *that*
   subset" and nothing about the private set, which is what ranks.
2. **The penalty CONTAMINATES every judgement about edges and
   divisions**, because node count and edge quality are multiplied
   together inside `adj_edge_jaccard`. A change that improved the edges
   and a change that merely improved the node count are indistinguishable
   in the only column this project has been reading.

**And they are separable — the metric already reports them apart.**
`METRIC_COLUMNS` carries `edge_jaccard` (raw J), `total_node_ratio`
((N_pred − N_true)/N_true) and `adj_edge_jaccard` as three columns.
**Every sweep in this log read only the third.** All future sweeps must
report all three.

### The direction this opens: TIGHTEN the gate, which nobody has tried

Every threshold sweep so far went DOWN (0.15 → 0.12 → 0.10 → 0.08).
**0.18, 0.22, 0.26 have never been run.** And the asymmetry favours it:

| | local (ratio 0.977) | hidden (inferred: at/over target) |
|---|---|---|
| loosen | edges better **+** penalty better → **helped twice** | edges better **−** penalty worse → cancels |
| **tighten** | edges worse **−** penalty worse → **hurt twice** | edges worse **+** penalty better → cancels |

**Tightening is the only direction where looking bad locally is
compatible with scoring well on the leaderboard** — the mirror image of
the two failures already measured.

## 2026-08-26 -- ⭐⭐⭐ thr4: on D187 the gate SURVIVES the rule, 44b6's dissent was a D32 artefact, and the two knobs are ADDITIVE

`biohub-thr4`, 22,911 s = 6.4 h, 6 arms × 3 domains. **Both anchors held
exactly**: `g0.15_s6` reproduced post_sweep's BASE (0.9254 / 0.8499 /
0.7992) and `g0.15_s4` reproduced short_sweep's short_4 (0.9279 / 0.8553
/ 0.8037).

| config | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| gate 0.12, `SHORT_MIN` 6 | +0.0033 | **+0.0006** | +0.0080 |
| gate 0.10, `SHORT_MIN` 6 | +0.0024 | +0.0027 | +0.0086 |
| `SHORT_MIN` 4, gate 0.15 | +0.0025 | +0.0054 | +0.0045 |
| **both** | **+0.0055** | **+0.0045** | **+0.0121** |

**thr3's 44b6 dissent (−0.0006) was a property of the D32 ranker, not
the data.** On D187 it is +0.0006. The ranker changed and the sign
flipped, which is why measuring on a model you do not ship is expensive.

**The two knobs are ADDITIVE**, unlike `gap2`'s, which were substitutes:

| | predicted if additive | measured |
|---|---|---|
| 6bba | 0.0033 + 0.0025 = 0.0058 | 0.0055 |
| zebrahub | 0.0080 + 0.0045 = 0.0125 | 0.0121 |

Two crossed experiments, two opposite answers. That is exactly why
independence cannot be assumed — recorded in `working-agreement.md`.

⚠ **This run has TWO independent domains, not three.** Every 44b6
dataset is inside D187's training pool, so that arm reports fit rather
than transfer. The log says so at the top.

## 2026-08-26 -- ⭐⭐⭐⭐ gap2: the CROSS was right, the gap line is dead, and the BASE anchor caught my own bug.

`biohub-gap2`, 204 min CPU, 10 arms × 3 domains. Built from the user's
"红色有很多是上一帧还检测到了,下一帧就检测不到了" and **crossed with
`SHORT_MIN` at their instruction**, against my original one-at-a-time
design.

### ⚠ First: the BASE anchor did NOT hold, and the cause was mine

| | recorded | gap2's BASE |
|---|---|---|
| adj_J | 0.9254 | **0.9253** |
| div_J | 0.1316 | **0.1282** |

The rewritten `_gap_close` normalised the gate by `step = span + 1`:

```python
D = D / float(step)      # span 1 -> step 2 -> gate 9 um becomes 18
```

**At span 1 — the shipped configuration — it divided by 2 and silently
doubled `GAP1_GATE_UM`.** BASE was not BASE. Fixed to divide by `span`,
which is 1 at span 1 and therefore byte-identical to the shipped code.

The docstring had claimed *"with GAP_SPAN=1 and GAP_NEED_CAND=True this
must be byte-identical … that is the BASE arm's job to prove"*, and the
anchor did its job. **What the anchor could not do is stop the run**, so
read the table below with the flaw in mind:

- **the `SHORT_MIN` deltas are unaffected** — the same loosened gate sat
  under both halves and cancelled in the difference, which is why
  `sm4_base` reproduced `short_sweep`'s `+0.0025 / +0.0054 / +0.0045` to
  four decimals;
- **the GAP arms are contaminated**, because they compare different
  spans and therefore different gates (×2, ×3, ×4 instead of ×1, ×2, ×3).

### THE CROSS WAS RIGHT: every gap arm's gain halves when `SHORT_MIN` goes 6 → 4

Each half is a delta against **its own** baseline.

| gap arm | 6bba sm6 | 44b6 sm6 | zh sm6 | | 6bba sm4 | 44b6 sm4 | zh sm4 |
|---|---|---|---|---|---|---|---|
| nocand1 | +0.0018 | −0.0016 | +0.0013 | → | **+0.0010** | **−0.0008** | **+0.0005** |
| span2 | +0.0002 | −0.0001 | +0.0003 | → | +0.0003 | −0.0001 | +0.0003 |
| span2i | +0.0019 | −0.0016 | +0.0013 | → | +0.0011 | −0.0008 | +0.0005 |
| span3i | +0.0019 | −0.0016 | +0.0013 | → | +0.0011 | −0.0008 | +0.0005 |

**Five arms, three domains, fifteen cells — every one shrinks by roughly
half.** The gap repair and the short filter are substitutes, exactly as
the flicker mechanism predicts: a wider bridge leaves fewer fragments,
so there is less for `SHORT_MIN` to delete.

**A one-at-a-time sweep would have overstated the gap arms by about 2x**,
and the user called it before the run. Record this as the standing
reason to cross knobs whose mechanisms touch: *the assumption of
independence is itself a claim, and here it was false.*

### The line is dead anyway

**Every gap arm fails the three-domain rule**: 44b6 negative in all of
them, 6bba and Zebrahub positive in all of them. And three structural
findings make it dead rather than merely unproven:

1. **`span2i` and `span3i` are identical in every column.** Bridging
   three frames finds nothing that bridging two does not.
2. **`span2` — the honest widening, candidate still required — is
   +0.0002 to +0.0003, i.e. nothing.** Exactly what `flicker.py`
   predicted: 79.8% of missed frames have no candidate in the repair
   band, so requiring one at *both* bridged frames fires on about
   0.2 × 0.2 = 4% of dropouts.
3. **`nocand1` ≈ `span2i` ≈ `span3i`.** All of the movement comes from
   dropping the candidate requirement at span 1. **The span never
   mattered at all.**

And pure interpolation costs divisions: div_J −0.0032 in `6bba` and
−0.0025 in Zebrahub. Inserting nodes the detector never saw manufactures
exactly the kind of node the division patch then mines for forks.

**Not re-running.** The fixed gate would loosen every arm equally, and
44b6's dissent is present at both `SHORT_MIN` values and in all five
arms — a looser gate does not flip a sign that consistent. The only
thing that survived is `sm4_base`, which is already submitted.

### Colab, for the record

The acceptance test did not finish again, and this time the session is
genuinely gone: **zero assignments on the server and no keep-alive
process**, unlike the three earlier cases where the VM was provably
alive. No theory offered — the evidence does not support one. One
fact worth keeping: **`hold` does not actually hold.** Its background
task reported "completed" while the keep-alive lingered for a further
three hours and then died with it.

## 2026-08-25 -- ⭐⭐⭐⭐ thr3: the detection gate FAILS the three-domain rule — but the dissenting domain is 44b6, by 0.0008, against Zebrahub's +0.0080.

`biohub-thr3`, 179 min CPU. Correctly loaded **D32** (`edge_ranker_d32.pkl`,
pool 32) and excluded the 16 `44b6` datasets inside that pool, because
this sweep changes DETECTIONS and the ranker-leak rule in
`working-agreement.md` requires it. **Its baseline is therefore
adj_J 0.9222, which lands 0.0002 from `biohub-div-gate`'s recorded D32
row — the anchor holds.**

⚠ **These absolute numbers are NOT comparable to `post_sweep`,
`short_sweep` or the submission**, all of which ran on D187. Only the
deltas within this run mean anything.

### Absolute

| gate | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| **0.15** (shipped) | 0.9222 | **0.8603** | 0.7988 |
| 0.12 | **0.9243** | 0.8598 | 0.8068 |
| 0.10 | 0.9236 | 0.8596 | 0.8074 |
| 0.08 | 0.9236 | 0.8594 | **0.8075** |

### Delta vs the shipped 0.15

| gate | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| 0.12 | +0.0021 | **−0.0006** | **+0.0080** |
| 0.10 | +0.0014 | −0.0008 | **+0.0085** |
| 0.08 | +0.0014 | −0.0009 | **+0.0087** |

**The rule says discard, and the rule was pre-registered, so discard is
the verdict.** But the shape must be recorded accurately, because the
script's own pre-written verdict string says *"6bba artefact"* and
**that is wrong about which domain dissents.**

- `6bba` gains and `zebrahub` gains — and Zebrahub gains **four times
  more**, +0.0080 against +0.0021. Zebrahub is the arm with dense
  ground truth, so its adj_J is over all cells, not over annotated ones.
- **`44b6` is the lone dissenter, at −0.0006 to −0.0009** — at or under
  the resolution floor.

So this is not one domain carrying a result the others refuse. It is two
domains agreeing, one of them strongly, against a third that moves by
less than the noise. That is a materially different situation from
`poolsize` (6bba alone, +0.0073, LB −0.001), and the record should not
file it as the same thing.

**Not acting on it.** Relaxing a pre-registered rule the first time a
result sits just outside it is how the rule stops meaning anything. The
honest state is: contested, with the dissent smaller than the floor, and
the decision belongs to the user.

### The column that explains the mechanism, and a new interaction

| gate | nodes | repair pool |
|---|---|---|
| 0.15 | 548,794 | **111,937** |
| 0.12 | 596,529 | 64,202 |
| 0.10 | 621,292 | 39,439 |
| 0.08 | 639,878 | **20,853** |

**Exactly conserved**: 0.15 → 0.08 adds 91,084 nodes and removes 91,084
from the repair pool. Lowering the gate does not create detections, it
**moves them from `_gap_close`'s repair pool into primary nodes** — the
script was written to print both precisely so a fall could not be
misattributed.

**And that collides with the flicker finding.** `_gap_close` repairs a
dropout by snapping to a candidate in `[CAND_THR 0.05, gate)`. At gate
0.08 that band holds 19% of what it held at 0.15. So **lowering the gate
progressively destroys the repair mechanism**, and `biohub-gap2`'s arms
— which depend on that pool — would behave differently under a lowered
gate. The two lines are not independent and a future sweep should cross
them, the same way `gap2` now crosses span with `SHORT_MIN`.

## 2026-08-25 -- ⭐⭐⭐⭐⭐ THE MISSES FLICKER. No annotated cell is permanently invisible, and 48% of dropouts are longer than the repair can reach.

**The user's observation, from watching the rendered movie**: in
`6bba_3abfe10a` many red circles are cells detected in one frame,
missing in the next, some flickering back and forth. `flicker.py`
measured it over 379 annotated GT tracks in the 12 `6bba` holdout
datasets, following the GT's own edges so a track is a real cell's
history.

| | tracks | |
|---|---|---|
| detected in EVERY frame | 270 | 71.2% |
| **detected in NO frame** | **0** | **0.0%** |
| INTERMITTENT | 109 | 28.8% |

**The "permanently invisible" class is empty.** So the ceiling
argument — dim is dim, nothing to do — does not apply to these losses.
Every annotated cell is found in some frame.

### The dropouts, by length

| length | runs | | |
|---|---|---|---|
| **1 frame** | 102 | **52.0%** | the only length `_gap_close` bridges |
| **2 frames** | 44 | 22.4% | unrepaired |
| 3 frames | 23 | 11.7% | unrepaired |
| 4+ frames | 27 | 13.8% | unrepaired |

**48% of dropouts are beyond the reach of the existing repair.**

### This is the same phenomenon as `SHORT_MIN`, seen from the other end

A two-frame dropout splits a track in two. If either piece falls under
6 nodes, the shipped `SHORT_MIN = 6` deleted it. That is why lowering it
to 4 gained adj_J in all three domains today — the filter was deleting
the fragments of real cells that flicker had shattered.

### What kills the obvious fix, and why the follow-up has TWO variables

`_gap_close` requires the inserted node to land on a real candidate from
the repair band `[CAND_THR 0.05, UNET_THRESH 0.15)`. But at the 465
missed frames:

| | |
|---|---|
| **no candidate within 5 µm at all** | **79.8%** |
| there is one | 20.2%, median score 0.1226, p75 0.1352 |

**Four times in five the heatmap response does not dip below threshold,
it vanishes.** So the existing repair can fire on at most a fifth of
dropouts, and a two-frame bridge needing a candidate at *both* missing
frames would fire on roughly 0.2 × 0.2 = 4%.

The remaining fifth sits just under 0.15, which is where
`biohub-thr-sweep`'s winning gate of 0.12 lands — those two results are
looking at the same cells.

### `biohub-gap2`, queued

`_gap_close` rewritten with two parameters, **crossed with `SHORT_MIN`
at the user's instruction** because the two are the same phenomenon and
cannot be assumed independent — if a wider bridge repairs the dropouts,
there are fewer fragments and `SHORT_MIN`'s optimum moves.

- `GAP_SPAN` ∈ {1, 2, 3} — how many missing frames to bridge. The gate
  is divided by the span so a longer bridge gets a proportional travel
  budget; without that it would test the gate, not the span.
- `GAP_NEED_CAND` ∈ {True, False} — whether an inserted node may be
  purely interpolated, with no detection evidence. The case for it: the
  official metric matches nodes by centroid distance at `max_distance`
  7.0 µm, so an interpolated node near a real GT cell counts as a match
  and brings two correct edges. The case against: interpolating across a
  genuinely ended track manufactures false nodes, and the metric carries
  a node-count penalty.

10 arms × 3 domains, with **two** must-reproduce anchors: `BASE`
(span 1, candidate required, `SHORT_MIN` 6) must give `biohub-poolsize`'s
D187 row, and `sm4_base` must give `biohub-short-sweep`'s short_4
deltas. `post_sweep` swept this function's four constants and all four
were noise — but the span is not a constant, it is the shape of the
loop, and nobody had ever moved it.

## 2026-08-25 -- ⭐⭐ depth3: the depth hypothesis does NOT survive. Signs disagree and nothing is significant.

Pre-registered before the run, and this is the branch that fired: *"recall
flat means the front/back difference on screen is not depth, and this
line closes with a negative."*

| domain | corr(recall, z) | p |
|---|---|---|
| 6bba | −0.6837 | 0.062 |
| 44b6 | −0.6800 | 0.064 |
| **zebrahub** | **+0.5011** | 0.206 |

Two domains fall with depth, **Zebrahub rises**, and at 8 bins the 5%
threshold is |r| ≥ 0.707 so **none of the three is significant**.
Zebrahub is the arm whose GT is dense, i.e. the only one whose `recall`
is recall over all cells rather than over annotated cells.

### The brightness link is broken in the opposite direction

| | r | p | |
|---|---|---|---|
| 6bba: GT brightness ~ z | **+0.8048** | 0.016 | **significant** |
| 6bba: recall ~ GT brightness | **−0.7788** | 0.023 | **significant** |
| 44b6: GT brightness ~ z | −0.0059 | 0.989 | nothing |

In `6bba`, **GT-point brightness RISES with depth** — the opposite of
attenuation — and brighter bins have *worse* recall.

### ⚠ Do not read that against the recorded 0.52/0.74

The recorded finding (recall 0.52 for the dimmest GT-brightness quartile
against 0.74 for the brightest) is **per cell**. The −0.7788 here is a
correlation over **8 bin means**. Those are different populations, and a
bin-level correlation can invert the individual-level one. **They are
not in contradiction; they are not the same quantity.**

That is the day's recurring error appearing inside my own analysis
design this time — the eleventh instance. A per-cell version of this
measurement would be the honest comparison and has not been run.

### A gap in the design, stated rather than glossed

Zebrahub's brightness column is `nan` throughout: the sampling only runs
where a raw volume is mounted, which is the two competition arms. **The
load-bearing arm has no brightness data**, so the depth → dim link is
tested only where the GT is sparse.

### What it buys

The depth line closes with a negative, and that protects the flicker
result above: the intermittent misses are **not** explained by cells
being too deep to see, which is the reading that would have made them a
ceiling rather than something to repair.

## 2026-08-25 -- ⭐⭐⭐⭐⭐ short_sweep: a CURVE, not a spike. `SHORT_MIN` 6 → 4 SUBMITTED as the first test of the three-domain rule.

`biohub-short-sweep`, 168 min CPU, 7 values × 3 domains. **Both anchors
held**, and the second one exactly: `short_4`'s deltas reproduced
`biohub-post-sweep` to four decimals in all three domains
(+0.0025 / +0.0054 / +0.0045). **The pipeline is deterministic and
post_sweep's table was real, not noise.**

### The curve

| `SHORT_MIN` | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| 1 | 0.9258 | 0.8552 | **0.8060** |
| 2 | 0.9258 | 0.8552 | **0.8060** |
| 3 | 0.9258 | 0.8552 | **0.8060** |
| **4** | **0.9279** | **0.8553** | 0.8037 |
| 5 | 0.9277 | 0.8533 | 0.8015 |
| 6 (shipped) | 0.9254 | 0.8499 | 0.7992 |
| 7 | 0.9263 | 0.8515 | 0.7968 |

**1, 2 and 3 are byte-identical in every domain**, so the solved graph
contains no components of length 1 or 2. The filter only ever acts on
lengths 3, 4 and 5.

### Restated as "which track lengths should be deleted"

That is the form the three-domain rule can actually judge:

| delete tracks of length | 6bba | 44b6 | zebrahub | verdict |
|---|---|---|---|---|
| 5 | −0.0025 | −0.0034 | −0.0023 | same sign — **harmful everywhere** |
| 4 | −0.0002 | −0.0020 | −0.0022 | same sign — **harmful everywhere** |
| 3 | +0.0021 | +0.0001 | −0.0023 | **signs disagree — contested** |

The shipped `SHORT_MIN = 6` deletes all three lengths, and two of those
three deletions lose in every domain. `SHORT_MIN = 4` stops deleting 4s
and 5s and keeps deleting 3s; whether to stop deleting 3s as well is
exactly what the rule cannot decide, so the value stops at 4.

4 beats 5 in all three (+0.0002 / +0.0020 / +0.0022) and beats the
shipped 6 in all three (+0.0025 / +0.0054 / +0.0045).

### SUBMITTED, and what is actually being tested

`biohub-submit-ranker` v5. **Exactly one non-comment line differs from
the deployed version**, verified by diffing against the source pulled
back from Kaggle (which was byte-identical to the local copy):
`SHORT_MIN = 6` → `4`.

**The comparison is against ref 55757064, LB 0.920 — NOT 0.921.** The
deployed kernel prefers `edge_ranker_d187.pkl` and refuses the older
one, so it is the D187 pipeline, and D187 is also what `post_sweep` and
`short_sweep` measured on. Comparing the result to 0.921 would move two
things at once, which is the error this log already records twice.

**What is being measured is the rule, not the constant.** The weakest
domain's +0.0025 is the same scale as the two local gains that failed to
appear on the leaderboard (windowing +0.0019 → −0.001; ranker pool
+0.0073 → −0.001). Both of those were measured on `6bba` alone. This is
the first candidate the three-domain rule has ever produced, so:

- **it appears on the LB** → three-domain agreement buys something a
  single-domain holdout does not, and the rule is worth its cost;
- **it does not** → the rule is not sufficient either, and no local
  measurement at this scale should drive a submission again.

Either answer is worth more than the constant.

### A correction the user made, recorded because I repeated the error

I priced this submission at "about 11.8 h of GPU quota" three times.
**A submission does not cost GPU quota.** The 11.8 h is the hidden-test
runtime against the competition's 12 h cap — a timeout risk, not a
budget charge. Treating it as a charge inflated the apparent price of
the single most informative experiment available. Recorded in
`kaggle-toolchain.md` section 9.

## 2026-08-25 -- ⭐⭐⭐ nms4, the density test in all three domains: 2 of 3, none significant. The line closes, and the interesting number is not about NMS.

`biohub-nms4`, 13 min CPU. True cells per frame in every domain —
counted per frame on Zebrahub, and from the organisers'
`estimated_number_of_nodes / T` on the competition arms. All three
density ranges are about 10x wide, so all three are measurable and the
`NO RANGE` guard did not fire.

### The table

| domain | sets | density range | **corr(lost to NMS, true density)** | corr(reachable BEFORE NMS, true density) |
|---|---|---|---|---|
| 6bba | 12 | 50–521 (10.3x) | **−0.0633** | **−0.5580** |
| 44b6 | 12 | 74–786 (10.6x) | **+0.4856** | +0.0311 |
| zebrahub | 8 | 51–562 (11.1x) | **+0.4067** | −0.0266 |

### Read the sample size first

**Not one of the six correlations reaches conventional significance.**
At n=12 the two-tailed 5% threshold is |r| ≥ 0.576; at n=8 it is 0.707.

| | r | p |
|---|---|---|
| 6bba before~density | −0.5580 | 0.059 |
| 44b6 lost~density | +0.4856 | 0.110 |
| zebrahub lost~density | +0.4067 | 0.317 |
| the other three | ≤ 0.064 | ≥ 0.84 |

Twelve datasets is twelve points. A correlation of +0.49 on twelve
points is a shape, not a fact.

### The verdict the pre-registered rule gives

**The density hypothesis does NOT survive.** Two domains show it at
about +0.45, `6bba` shows nothing at all (−0.06, p=0.85). Signs
disagree, so by the rule written down before the run it is a property of
some domains and not of the method.

This was the user's line — they pushed three times to have it tested
where cells are actually crowded, against my closing it on `6bba` alone.
Testing it properly was right; the answer is that it does not replicate.

### The one number worth keeping is not about NMS

`6bba`'s **−0.5580 (p=0.059)** says that in `6bba`, datasets with more
true cells per frame have systematically worse reachability **before any
suppression happens**. The detector is failing upstream. The other two
domains show nothing there (+0.031, −0.027).

That is a candidate explanation for why `6bba` shows no NMS density
effect — the crowding damage is already done before suppression runs, so
suppression then takes an even share. **It is a hypothesis, not a
result**, and it is one point short of significance. Carried as
unresolved, per the standing rule about not explaining away a
disagreeing measurement.

If it holds it matters more than NMS ever did, because **the 12-dataset
holdout this project measures everything on is entirely `6bba`** — the
one domain with a density-dependent detection problem.

Caveat on it: `6bba` reachability is against annotated cells (9.71%).
The correlation is across 12 datasets within one domain, so the
annotation convention is roughly constant — but "roughly" is doing work
and has not been checked.

### The radius grid, refined, and why it closes the line

nms3's grid put everything between two adjacent points. Refined to
(2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5), the curve is **smooth and
monotone in all three domains** — no plateau, no kink, no natural
optimum. Ceiling gain from the shipped 4.0 down to 3.0:

| domain | 4.0 | 3.0 | gain | extra points kept |
|---|---|---|---|---|
| 6bba | 0.9561 | 0.9564 | +0.0003 | +645 |
| 44b6 | 0.9527 | 0.9535 | +0.0008 | +1720 |
| zebrahub | 0.8947 | 0.8972 | +0.0025 | +1246 |

**Reachability is monotone in the radius and therefore structurally
incapable of choosing one** — it always votes smaller, all the way to
keeping every duplicate peak. Only the official score, which pays for
the extra points through precision and the node-count penalty, can pick.

And the ceiling gain is +0.0003 and +0.0008 in two of three domains,
at or under the resolution floor, **before** paying for the extra
points. A several-hour full-pipeline radius sweep would be spending
that time to chase a number the ceiling already says is too small.
**The NMS radius line closes here**, with the measurement the user asked
for rather than the one-domain version I tried to close it on.

## 2026-08-25 -- nms3 PARTIAL: two sparse arms landed, the dense arm crashed, and the two sparse arms CANNOT be compared to each other

`biohub-nms3` v1 ran `6bba`, ran `44b6`, then died entering Zebrahub.
**Both sparse arms landed and the 6bba anchor held** (0.0092 here against
`biohub-nms-loss`'s recorded 0.0091).

### What suppression costs, per domain

| domain | cells/frame | annotated | before | after | **cost** |
|---|---|---|---|---|---|
| 6bba | 97 | 9.71% | 0.9652 | 0.9561 | **0.0092** |
| 44b6 | 327 | **0.77%** | 0.9581 | 0.9527 | **0.0054** |

### ⚠ The cross-domain reading is NOT available, and the first version of this entry made it anyway

The first version of this entry said *"this is the opposite of the
density hypothesis — the denser domain loses less."* **Withdrawn.** The
user caught it within minutes, on the grounds that recall and annotation
are not the same thing.

`reachable()` counts **GT nodes with a detection point within 5.0 µm**.
Its denominator is not the cells that exist, it is **the cells somebody
chose to label**. So 0.9581 does not read "95.81% of 44b6's cells", it
reads *"95.81% of the two-or-three cells per frame that were annotated
in 44b6"*. That is a recall against a hand-picked sample, and it equals
recall only if annotation is a random draw from the true cells — which
nobody does, because annotators label what they can see.

The two rows therefore have **differently-selected denominators**:
`6bba` keeps 9.71% of its cells, `44b6` keeps 0.77%, a 12x stricter
sieve. If annotation favours bright isolated cells — and Zebrahub's
recall-by-brightness-quartile of **0.52 dimmest vs 0.74 brightest** says
brightness dominates detectability — then `44b6`'s sample is the more
purified of the two, and *isolated* is exactly the property that makes a
cell safe from suppression. **0.0054 < 0.0092 is consistent with 44b6
suppressing more gently AND with 44b6 being annotated more selectively.
These two rows cannot separate them.**

What survives: the *within-domain* differences are clean, because before
and after are measured on the same annotated set. "Suppression costs
0.92% of 6bba's annotated cells" stands. "0.0092 vs 0.0054 across
domains" does not.

Same error class as the six others today — **numbers from different
populations compared to each other** — and this one appeared in the same
entry that warned both arms were sparse.

### Why this makes the Zebrahub arm the whole experiment

Zebrahub labels **every** cell. Only there is the denominator the true
cell population, only there is the word "recall" earned, and only there
can the number be compared to anything else. It is the arm that did not
run. v2 is queued.

The radius sweep has the same limit and is recorded for later use, not
read now: on `44b6`, 4.0 → 2.0 moves reachable-annotated from 0.9527 to
0.9535, flat below 3.0.

### The crash

```
T = max(int(k[1:]) for k in z.files if k.startswith("p")) + 1
ValueError: invalid literal for int() with base 10: 'arent_track_id'
```

`prenms2` writes the peaks as `p0…pN` / `q0…qN` **and** puts the ground
truth in the same file, including `parent_track_id` — which also starts
with `"p"`. `startswith` alone matched it. The prefixes `p`/`q` were
chosen deliberately to avoid colliding with the detection cache's
`c`/`s`, and then collided with a GT array instead.

Fixed in v2 by a shared `_frame_keys(files, prefix)` that requires the
remainder to be digits, applied to **all three** frame-key scans rather
than only the one that broke, and unit-tested against the real key set.

Same family as `arm.lower()` on a float and `prev_of` as a dict: the
name existed, the arity matched, nothing static could see it.

## 2026-08-25 -- ⭐⭐⭐⭐ THE POST-LINK SWEEP: one of fourteen survived. `SHORT_MIN` 6 → 4, positive in all three domains.

`biohub-post-sweep`, 245 min, CPU. Seven post-link constants inherited
from the public solution, fourteen one-at-a-time changes, scored in
`6bba` (12 holdout), `44b6` (12 outside D32's pool) and Zebrahub (8
crops). Both anchors held: BASE `6bba` reproduced `biohub-poolsize`'s
D187 exactly at score 0.9386 / adj_J 0.9254 / div_J 0.1316.

### BASE, the shipped configuration

| domain | score | adj_J | div_J |
|---|---|---|---|
| 6bba | 0.9386 | 0.9254 | 0.1316 |
| 44b6 | 0.8499 | 0.8499 | **0.0000** |
| zebrahub | 0.8014 | 0.7992 | 0.0221 |

### The result

**`short_4` is the only change that survived the pre-registered
three-domain rule with a delta big enough to see.**

| config | 6bba | 44b6 | zebrahub | verdict |
|---|---|---|---|---|
| **short_4** (`SHORT_MIN` 6→4) | **+0.0025** | **+0.0054** | **+0.0045** | **same sign, positive** |
| short_8 (`SHORT_MIN` 6→8) | +0.0007 | +0.0044 | **−0.0056** | signs disagree |
| lf_w_0.5 | −0.0015 | −0.0027 | −0.0032 | same sign, negative |
| lf_win_1 | −0.0002 | −0.0007 | −0.0020 | same sign, negative |
| gap_minsc_0.15 | +0.0002 | −0.0003 | −0.0008 | signs disagree |
| lf_w_1.0 | +0.0014 | −0.0019 | +0.0007 | signs disagree |
| lf_win_3 | +0.0010 | −0.0027 | +0.0004 | signs disagree |
| the four `gap_*` gate/snap/cap rows | ±0.0000 | ±0.0006 | ±0.0001 | noise, see below |

### What `SHORT_MIN` is, in plain terms

`_short_filter` deletes a whole predicted track when it has fewer than
`SHORT_MIN` nodes, unless that track contains a division. The shipped
value is 6. Lowering it to 4 means **keeping the 4- and 5-node tracks we
were throwing away**, and doing that gains adj_J in every domain.

The filter is a bet that short predicted tracks are mostly wrong. At
length 4–5 the bet loses: those fragments are mostly right, and deleting
them removes correct edges from the numerator. Going the other way
(`short_8`, delete even more) helps `44b6` and hurts Zebrahub — that is
the signature of a domain-specific effect, and the rule discards it.

### A correction to the script's own verdict column, made here so it is not repeated

`gap_gate_7` printed **"SAME SIGN, POSITIVE"** with three deltas of
`+0.0000`. That is not a win. All three were positive but under `1e-4`,
and the test as written had no lower band, so it promoted a number a
thousand times smaller than the holdout's own noise. **Agreeing in sign
is not the same as being large enough to see.** `short_sweep.py` adds an
explicit `TINY = 5e-4` floor; anything whose weakest domain is under it
prints `TOO SMALL TO MATTER` instead. Read `short_4` as the only real
survivor of the fourteen.

### The one column that carries no information

`44b6`'s BASE div_J is **0.0000** — the 12 `44b6` datasets in this arm
produce no correct divisions at all, so every `44b6` entry in the div_J
delta table is `+0.0000` by construction. It is not evidence that these
changes leave divisions alone; it is evidence that there is nothing
there to move.

### What happens next

1. **`biohub-short-sweep` is queued** (CPU, 7 configs × 3 domains):
   `SHORT_MIN` ∈ {1, 2, 3, 4, 5, 6, 7}, where 1 means the filter is
   switched off entirely. **Is `short_4` a curve or a spike?** A real
   effect varies smoothly with the constant; a lone value that wins
   while both neighbours lose is what noise looks like. It carries two
   must-reproduce anchors, the second being `short_4`'s three deltas to
   four decimals — same code, same inputs, so any drift means the
   pipeline is not deterministic and this whole table was noise.
2. **Then, and only then, the leaderboard.** `SHORT_MIN` is line 127 of
   `submit_ranker.py`, the script behind ref 55707206 (LB 0.921), so
   `6 → 4` is a genuine single-variable submission. That is the
   experiment that tells us whether **the three-domain rule is worth
   anything** — the local holdout is 0 for 2 against the leaderboard,
   but both of those failures were `6bba`-only. This is the first
   candidate the rule has ever produced. Costs ~11.8 h GPU and one
   submission slot; **needs the user's go-ahead** because it is
   outward-facing.

## ⭐ PRIORITY as of 2026-08-25 (user's call): DIVISIONS FIRST, and the reason is structural

**This overrides the 2026-08-24 ordering below.** It is not a change of
mind about the loss budget -- that table is still correct. It is that
morphology arriving changed what the division machinery IS.

The user's argument, in their words: divisions were being caught by the
ORPHAN FORK, but morphology features are EDGE-correlated, so the simple
orphan fork can no longer serve as the division criterion.

Unpacked, and it is sharper than the "orphan starvation" worry recorded
in `morph_edge.py`:

**The orphan patch was never a division detector. It is a LINKER-FAILURE
detector.** Its raw material is the ILP refusing the second daughter and
leaving her unlinked; `_add_safe_divisions` then requires a parent with
EXACTLY ONE child (`div_train.py:418`, `if len(ch) != 1: continue`).
What it catches is therefore a function of how the linker fails. Change
the linker -- which is exactly what putting morphology in the ranker
does -- and what it catches changes, in a direction the division side
cannot predict.

Two regimes, opposite results, and the patch is silent in both:

| ranker becomes | what the ILP does | orphan patch | div_J |
|---|---|---|---|
| more FASTIDIOUS (daughters "look different", edges pushed down) | refuses both daughters, `len(ch)==0` | never fires | falls |
| more CORRECT | takes both daughters, `len(ch)==2` | never fires | rises |

### The third regime, and it is the one already running (user, 2026-08-25)

The two rows above are the cases where the patch goes SILENT. The user
named the dangerous one: the patch FIRES, on things that are not mother
and daughter at all. It is not hypothetical -- it is the status quo,
measured on the 12-dataset 6bba holdout by `biohub-poolsize` and carried
in ref 55757064's description:

| ranker | div TP | div FP | div FN |
|---|---|---|---|
| D32 | 3 | **21** | 9 |
| D187 (submitted) | 5 | **26** | 7 |

**26 false divisions bought 5 real ones.** Two things keep that from
being fatal, and both are already on record:

- a fork lying entirely among unannotated cells enters no FP set
  (user's observation, verified): 913 extra forks cost -0.0027 adj_J,
  about 30 visible edge FPs, so ~97% of extra forks are INVISIBLE.
- but div_J is not blind to them, and div_J is the 0.1 term.

And the exchange rate cannot be improved by forking more: across all 12
`gate_sweep` points (forks 0 to 1847) **division FN never drops below
6** while division FP explodes. The divisions we miss are not in the
candidate set. So the only move with value left is KILLING FALSE FORKS,
not proposing more.

That is what mass conservation is for: a real division sums to one
mother, a mother-plus-passing-neighbour sums to two.

**Bound on `div_mass`:** its fake forks are built from GT (mother + true
successor + any GT cell within GATE_UM), which is an EASIER population
than the patch's actual false positives -- those already survived sister
distance, mutual-nearest, both-children-continue and divergence. A good
AUC there says the route is worth walking; it does not say the 26 fall.
Testing that needs the patch's real survivors.


`poolsize` already ran the second regime: a better ranker moved div_J
0.0909 -> 0.1316. That is on record and it is why "a more confident
linker drains the orphan pool" was too simple -- CORRECT and CONFIDENT
are different axes.

**The consequence: divisions need a criterion that does not depend on
the linker failing first.** Mass conservation is a candidate for exactly
that -- "these two cells sum to one mother" is answerable without the
ILP having made a mistake. `div_mass.py` -> `biohub-div-mass` is that
measurement. If it lands, divisions decouple from linking and the
2026-08-24 ordering can resume.


SUPERSEDED 2026-08-25 by the block above -- ## ⭐ PRIORITY as of 2026-08-24 (user's call): detection and linking, not divisions

Divisions are parked (item 1). The loss budget says why -- decomposed
from the det-threshold table below:

| where the score leaks | size |
|---|---|
| detection recall: a GT edge's endpoint was never detected | ~1.75% |
| linking: a reachable edge linked wrong | ~1.6% |
| false edges: predicted where GT has none | ~3.3% |

**The last two together are about 3x the first**, and adj_J carries 10x
div_J's weight. Concrete leads, cheapest first:

- **A. Which of the xl detector's 12 forward passes earn their place.**
  `biohub-det-ablate` RUNNING. Three branches x four flip views, never
  ablated. Branches 1 and 2 get PIXEL-IDENTICAL input (both top-hat), so
  their errors should correlate; branch 0 is the only different view of
  the data and has the worst solo `val_recall` -- which is not what earns
  a place in an ensemble. One inference pass covers 11 configurations.
- **B. The ranker never sees `edge_prob`.** The transplant dropped
  yusuketogashi's first-listed feature because `detcache_*.npz` carries
  only coords and detection scores (full entry below). NOT free: the
  cached probs index the OFFICIAL detector's nodes, the pipeline runs xl
  nodes, so it needs the official edge head re-run over the xl node set.
- **C. Multi-frame trajectory in LINKING.** The official edge head is
  `window_size = 2` and "structurally cannot use velocity"; our ranker
  has one speed scalar greedily estimated from a single transition. The
  same shape of gap the division work just proved real.

  Two public references, both already on the table and neither pursued:

  - **Togashi's item 4, acceleration lookahead in motion prediction, was
    never transplanted** (`public-solutions-recon.md:313`). Pure CPU,
    existing caches, and it is a 0.915 solution's own component. Cheapest
    thing on this list.
  - **HOCT -- higher-order cell tracking**, forum /726521, posted by
    **Jordão Bragantini as HOST**: he is a co-author and it is the Royer
    lab, i.e. the lab that made this data. Paper arxiv 2607.11754, code
    and PRETRAINED MODEL at `github.com/royerlab/hoct`. "Higher-order" is
    exactly the multi-frame motion constraint this item is about, and he
    says plainly "we haven't tried it on the competition dataset yet".
    **Training code is withheld (under peer review), so it is
    inference-only.**

    The one recorded attempt is NEGATIVE and its reason is instructive: a
    rank-24 competitor fed it **uniform-ball masks** and it over-linked
    into spurious forks (/728551, via `strategy.md`). Bragantini's own
    caveat is "results depend on the quality of the segmentation mask" --
    and our pipeline emits CENTROIDS, not masks. Uniform balls are the
    worst possible mask. **Untested idea: watershed the xl detection
    heatmap seeded at the detected peaks, which respects real cell shape
    and is strictly better input than spheres.** Submission is offline,
    so the weights would need packaging as a dataset.
- **D. The ASSIGNMENT step, not the scoring.** `ranker_scale` measured
  per-dataset ranker AUC FLAT against density while adj_J slides hard --
  so crowding damages the pick, not the score. Untouched.

Discipline, from two failures today: an offline AUC/AP win is not an
adj_J. Everything here gets re-measured through the full pipeline before
it is believed.

### Why this ordering is structural, not a preference (user, 2026-08-24)

Divisions sit DOWNSTREAM of both. The forensics funnel says it exactly:
of the 12 holdout GT divisions, **1 parent undetected + 1 daughter
undetected + 3 outside `GATE_UM` + 2 below `ILP_FLOOR` = 7 are lost
before the division mechanism gets a say**, and it already takes all 5
that reach it. No division model can recover what detection and linking
threw away first.

The orphan patch is the sharper case: it feeds on the LINKER'S FAILURES.
A second daughter becomes adoptable only if it was detected AND the 1:1
assignment left it unclaimed. That is why improving the ranker backfired
once already -- "1:1 Hungarian steals second daughters into wrong
continuations, **draining the orphan pool the division patch mines**"
(`ranker_fine.py` header). A linker that gets more confident without
getting more correct starves the mechanism downstream of it.

So detection and linking are not merely higher-value targets. They are
the precondition for the division term to be worth anything at all.

### Process notes, each from something that went wrong today

- **A static undefined-name pass catches real bugs. Do not push when it
  is dirty.** It flagged `gc` as undefined; the push went out anyway and
  wasted a run. It had earlier caught `class UNet3D` and `_dist_um`
  missing from an assembled script -- both would have died at runtime.
- **Holdout deltas below ~0.003 are not resolvable on the LB.** Measured:
  two submissions whose holdout differed by +0.0014 scored -0.001 apart.
  Much of this session's fine tuning (the div_w peak at 0.55 was 0.0018
  above its neighbours) sits under that floor and should be treated as
  noise.
- **The family-offset rule predicts to about +-0.003, not +-0.001.** Two
  points now: -0.0158 and -0.0182. The first prediction landing on the
  nose was luck.
- Kernel assembly by slicing one script's head off another accumulates
  dead code -- `div_train.py` carries `GATE_SWEEP`, `train_at_gate`,
  `funnel` and two copies of `GATE_XY/GATE_Z` that nothing calls.
  Rebuild it from minimal parts before extending it further.



- **LB best 0.917** (ref 55690436, verbatim xl reproduction). Holdout
  baseline for the xl family: 0.9318 (adj 0.9185 / div 0.1333);
  family-specific local-to-LB offset ~-0.015.
- **Edge re-ranker line (2026-08-23):** v1 failed (sparse-GT labeling
  gave 97% positives); v2 fixed labels by EXCLUSION negatives (the
  candidate hitting the GT child is positive, all other in-gate
  candidates of that source are negative) -> 39.8k pairs, 46.8% pos,
  AUC 0.9884, linker-only adj_J 0.9239 vs Hungarian 0.9185 (+0.0054);
  v3 full-pipeline composite only 0.9320 vs 0.9318 — the edge gain is
  eaten by div_J 0.1333->0.0800. **Diagnosis (user's): 1:1 assignment
  steals division second-daughters into wrong continuations, draining
  the orphan pool the division patch mines.** v4 (in flight) tests the
  reconciliation: the ranker's labels already score parent->second-
  daughter edges HIGH (both GT children were positives), so a
  capacity-2 ILP fed ranker probs (division surcharge 0.75/0.85/0.95)
  can take both daughters natively; plus Hungarian p_min 0.30-0.40.
  Detection caches for all 44 datasets + the trained ranker pkl live in
  the `biohub-edge-ranker` kernel output (iterations cost ~minutes).
- **Toolchain notes for the next session:** CLI needs
  `KAGGLE_API_TOKEN` minted per call (kaggle-toolchain.md 1b); pack
  wheels + `pip -e repo --no-deps` is the proven env for ILP+scoring
  kernels; ERRORed kernels' outputs are downloadable but NOT attachable
  as kernel sources; official metric needs a GPU machine even to score.
- Uncommitted: two days of docs + scripts (progress/reference/recon
  restructure, zebrahub_extract rewrite, finetune_detect v3). Commit
  when the user says so.

## 2026-08-25 -- NMS costs 0.9% ON 6bba. NOT closed: the density hypothesis was tested on the LEAST dense domain.

> ⚠ **CORRECTED by the user before the ink was dry.** The hypothesis was
> that a FIXED 4.0 um radius hurts where cells are CROWDED. This run
> measured it on `6bba` at **97 cells per frame** -- the least crowded
> domain available. `44b6` runs **327** and Zebrahub's crops span
> **45 to 809**. A null result on the sparsest data says almost nothing
> about a density hypothesis, so **"the radius is already right" is
> supported for 6bba and nowhere else.** `nms3.py` runs all three.
>
> Why it happened, since the mechanism matters more than the instance:
> `biohub-prenms-cache` was built BEFORE the three-domain standard and
> covered only the 12 `6bba` holdout, and I ran the experiment the data
> allowed instead of waiting for the data the question needed. **Running
> what is available rather than what answers the question is its own
> failure mode**, and it produced a confident "closed" that was not
> earned.
>
> The user also pointed at where the test actually belongs: **Zebrahub
> is the only domain whose GT is dense, so it is the only place TRUE
> cell density per crop is known** -- on competition data
> `corr(annotation density, true density) = -0.20`. With crops spanning
> 18x in density, the right measurement is NMS loss correlated against
> true density across them, not three isolated numbers.

`nms_loss.py` -> `biohub-nms-loss` v1, **9 min on CPU**, on
`biohub-prenms-cache`'s pre-suppression peaks for the 12 `6bba` holdout
datasets.

| | |
|---|---|
| GT reachable BEFORE suppression | **0.9652** |
| GT reachable AFTER suppression | **0.9561** |
| **cost of NMS** | **0.0091** |

Per dataset the loss runs 0.0% to 3.8%, mostly under 1%.

### The radius sweep has nothing in it

| rule | recall | detections kept |
|---|---|---|
| fixed 2.0 um | 0.9564 | 168,240 |
| fixed 3.0 um | 0.9564 | 168,220 |
| **fixed 4.0 um (shipped)** | **0.9561** | 167,575 |
| fixed 5.0 um | 0.9554 | 166,609 |
| fixed 6.0 um | 0.9519 | 163,411 |
| adaptive, beta 0.30 | 0.9564 | 168,205 |
| adaptive, beta 0.45 | 0.9561 | 167,657 |
| adaptive, beta 0.80 | 0.9320 | 155,454 |

Halving the radius buys **+0.0003 recall for 665 more detections**, and
the adaptive rule -- radius = beta x the frame's median
nearest-neighbour distance, which was meant to remove the tuned
constant -- does not beat the fixed one anywhere.

**On `6bba`, at 97 cells per frame.** See the correction at the top:
this is the domain where the density hypothesis predicts the smallest
effect, so the table above cannot close the question. The adaptive rule
in particular was designed to matter WHERE DENSITY VARIES, and 6bba
alone barely varies.

### ⚠ Two errors of mine this run exposes

**1. The density correlation is computed against the wrong quantity.**
The `cells/fr` column reads 5-13, which is ANNOTATION density, not true
cell density -- `reference.md` records `corr(annotation density, true
density) = -0.20`, i.e. **negative**, and "N_true cannot be inferred
from the GT". So `corr(fraction lost to NMS, cells per frame) = +0.0041`
says nothing about true density and must not be quoted as if it did.

The conclusion survives anyway, by MAGNITUDE rather than by
correlation: NMS removes 0.91% of reachable GT in total, so even if
every bit of it landed on the densest datasets it could not produce
`corr(adj_J, density) = -0.733`. **The recorded explanation -- that the
density penalty sits downstream of edge scoring, in near-ties among many
candidates -- still stands, and the NMS hypothesis is refuted.**

**2. `det_floor`'s "13.1% of cells have no peak at any threshold" is a
ZEBRAHUB number and I generalised it.** I had called the 9.5%
"has a peak but is discarded" the largest recoverable loss on the board
and pointed NMS at it. That reframing applied a domain-gap number to the
competition, and the lead it motivated is now measured and empty.

**3. And the correction in (2) was ITSELF a bad comparison** -- the
user asked which dataset the NMS recall came from, which exposed it:

| | value | what the percentage MEANS |
|---|---|---|
| Zebrahub | 0.8688 | **dense GT: 86.9% of ALL cells have a peak** |
| 6bba (competition) | 0.9652 | **sparse GT: 96.5% of the 9.71% that are ANNOTATED** |

**These are not the same measurement.** The 6bba figure says nothing
about the ~90% of cells nobody labelled, and there is reason to suspect
the labelled ones are easier: the Zebrahub bench measured recall by GT
brightness quartile at **0.52 for the dimmest quartile against 0.74 for
the brightest**. If competition annotation skews bright, 96.5% is
optimistic by an unknown amount.

So the honest statement is: **among annotated cells on 6bba, 96.5% have
a peak.** Whether detection is "not where the loss is" on the full cell
population is UNMEASURED, and cannot be measured where the GT is sparse.

### Which dataset each 2026-08-25 experiment actually ran on

Written down because this session has repeatedly compared numbers from
different populations, and every instance was caught by the user rather
than by me.

| experiment | data | GT |
|---|---|---|
| `nms_loss`, `thr_sweep`, `poolsize`, `node_filter`, `div_gate` | 12 `6bba` holdout | **sparse** |
| `post_sweep` | 12 `6bba` + 12 `44b6` + 8 Zebrahub | mixed |
| `det_floor`, `det_ablate3`, `tta_probe`, `node_bright`, `morph_*`, `div_mass` | Zebrahub crops | **dense** |

**A percentage measured against sparse GT is not comparable to one
measured against dense GT**, even when both are called "recall".

### What this leaves

Detection on competition data is not where the loss is: 96.5% reachable
before NMS, 95.6% after, and the shipped gate and radius are both at or
near their best. The remaining gap between 95.6% reachable and the
pipeline's actual performance is downstream -- in which candidate the
linker picks, which is where `corr(adj_J, density) = -0.733` already
pointed.

## 2026-08-25 -- the GT is 212 KB, and the 44b6 arm is 2x noisier than the 6bba one

`gt_export.py` -> `biohub-gt-export` v1, **1 min on CPU**. Writes what
`read_geff` returns as one npz per dataset, for the 12 `6bba` holdout
and 12 `44b6` outside D32's training pool.

**24 datasets, 13,279 nodes, 12,839 edges, 212 KB total.** The
competition data is 87.6 GB of zarr and its `.geff` directories cannot
be fetched file-by-file; the GT inside them is a fifth of a megabyte.
That unblocks any machine without the competition mount -- see
`kaggle-toolchain.md` section 11 for the Colab worker it was blocking.

(Degradation noted in the log and worth keeping: `zarr` was not
installed, so `read_geff` fell back to its filesystem reader. A `.geff`
IS a zarr store, so that path matters even though the `geff` library is
optional.)

### ⚠ The measurement this exposes, which bears on `biohub-post-sweep`

| domain | datasets | GT edges | per dataset |
|---|---|---|---|
| `6bba` | 12 | **10,313** | 859 |
| `44b6` | 12 | **2,526** | **210** |

**The 44b6 arm carries 24% of the 6bba arm's ground truth**, so its
adj_J estimate is about **2x noisier** (standard error goes as
1/sqrt(n)). Its thinnest dataset has **49 edges**.

**44b6 is the worst of both worlds for a sweep arm:** 327 cells per
frame against 6bba's 97, so the ILP costs the most there, while the
annotated fraction is 0.77% against 9.71%, so the measurement is the
weakest. That is also why `biohub-post-sweep` ran past three hours
against an estimate of 100 minutes -- the estimate used `poolsize`'s
67 s per dataset, **measured on 6bba only.**

### How to read the three-domain rule given this

The rule -- an effect counts only if its adj_J delta has the same sign
in `6bba`, `44b6` and Zebrahub -- controls false positives well and has
**low power**. If each domain independently shows the correct sign for a
marginal effect about 65% of the time, all three agreeing happens ~27%
of the time. Most real small effects will fail it.

That is defensible: the holdout is 0 for 2 against the leaderboard at
this scale, so an effect the rule cannot detect is one that would not
have been actionable. **But do not read "nothing survived the rule" as
"these constants are already optimal."** Those are different claims, and
only the first one will have been measured.

If power is wanted later, enlarge the 44b6 arm rather than loosening the
rule: 71 `44b6` datasets exist and 12 are being used. Using every one
outside D32's pool would roughly quadruple that arm's ground truth, at
the cost of runtime and no quota.

## 2026-08-25 -- ⭐⭐⭐⭐ THE LOCAL HOLDOUT IS 0 FOR 2 AGAINST THE LEADERBOARD

**Submission 55757064 (D187 ranker, pool 32 -> 187) scored LB 0.920
against the D32 baseline's 0.921.** The local holdout had it at +0.0073
score and +0.0032 adj_J, reproduced bit for bit by two independent runs,
with both anchors holding.

### State it precisely: not "worse", INDISTINGUISHABLE

| | 55707206 (D32) | 55757064 (D187) |
|---|---|---|
| LB public | **0.921** | **0.920** |
| `total_bytes` on the hidden test | 262,623,879 | 262,803,696 (**+0.07%**) |
| local holdout score | 0.9313 | 0.9386 (**+0.0073**) |
| local holdout adj_J | 0.9222 | 0.9254 (+0.0032) |

The LB reports three decimals, so 0.001 IS its reporting resolution, and
the two pipelines emitted output within 0.07% of each other on the
hidden test. **The honest reading is that they are the same submission
as far as the leaderboard can see** -- which is worse news than losing,
because the +0.0073 the holdout measured simply is not there.

### Both paired local-vs-LB comparisons we have ever run

| change | local delta | LB delta | hidden-test output |
|---|---|---|---|
| windowed vs global | **+0.0019** | **-0.001** | +10% |
| ranker pool 32 -> 187 | **+0.0073** | **-0.001** | +0.07% |

**Two for two.** Every time a positive local delta has been taken to the
leaderboard it has failed to appear. The second one is the larger and
better-anchored of the two.

### The signal we explained away

`biohub-poolsize`'s own entry recorded that **D187's edge AUC was LOWER**
(0.9864 against 0.9884) while its adj_J was higher, and filed it as "a
bigger pool means a bigger, harder validation split, so edge AUC cannot
be compared across configurations". That rationalisation may have been
wrong: the simpler reading is that the larger pool made the ranker
slightly worse and the local adj_J gain was specific to `6bba`. **When a
second measurement disagrees with the one you like, do not explain it
away; carry it as unresolved.**

### What this changes

- **The best submission is still ref 55707206, LB 0.921.** Final
  selection allows two; that one holds a slot.
- **A local gain under ~0.01 on the 12-dataset holdout is not evidence.**
  All twelve are `6bba`, one imaging batch, and many knobs have now been
  tuned against them.
- The user identified both failure modes before this result landed:
  researcher overfitting to a 12-dataset holdout, and "only the LB is
  truth, but that means a lot of GPU or CPU may be wasted". Both are now
  measured rather than argued.
- `biohub-post-sweep`'s pre-registered three-domain rule -- an effect
  counts only if its sign agrees across `6bba`, `44b6` and Zebrahub --
  was designed as a precaution and now has evidence behind it.

### What it does NOT license

It does not say the pipeline cannot be improved, and it does not say
local measurement is useless. Local work still reliably rejects things:
`node_filter`'s -0.03 and `morph_edge`'s -0.0015 are far outside any
noise and would not have been worth an LB slot. **The holdout screens
out; it does not select in.**

## 2026-08-25 -- multi-frame linking, all three attempts, and the one that was never recorded

The user asked what happened to multi-frame tracking. Three attempts
exist; **`biohub-accel`'s result was never written down** and survived
only in a session scratchpad, which is how a run gets repeated by
accident. Recorded here.

### 1. Acceleration lookahead (Togashi's item 4) -- `biohub-accel`

Five columns on top of the ranker's 18: `acc_len`, `acc_cos`, `md_acc`,
`md-md_acc`, `acc_hist`.

| arm | cols | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASE | 18 | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| ACC | 23 | 0.9319 | 0.9235 | 0.0833 | 3/24/9 |
| delta | | -0.0067 | **-0.0019** | -0.0483 | |

Stage A agreed in advance: edge AUC 0.9867 against 0.9864, permutation
importance +0.0000 to +0.0017 across all five.

**⭐ The cause is in the degradation counters, not in the model.**

```
accel_no_parent: 316,658
accel_one_step:  246,780
```

against **225,861 candidate pairs**. Acceleration needs THREE
consecutive positions; the backward chain is built by `pass1_tracks`, a
greedy linker, and for most candidates it cannot produce three. All five
columns are then filled with 0 -- which the model reads as "no
acceleration", not as "unknown".

**So this is not evidence that multi-frame information does not help. It
is evidence that this implementation could not compute it.**

Note also, by 2026-08-25's standard, that -0.0019 sits inside the range
where the holdout has inverted against the leaderboard twice. "It hurt"
is not supportable either. What is supportable: **no measurable effect,
and the feature is undefined for most rows.**

### 2. HOCT, higher-order cell tracking -- `biohub-hoct-score`

`similarity` AUC **0.7012** with images, against **0.9091** for plain
negative distance on the same 364 edges. Lost by 0.21. Training code is
withheld so there is no fine-tuning path. Closed; full entry below.

### 3. The division model's forward features -- `biohub-div-kinematics`

`sep_slope` separates a true division from a random pair 1.155 against
0.082, a 14x gap and the cleanest signal this project has found. It is
genuinely multi-frame. `biohub-div-gate` then wired it into the pipeline
and the prune sweep was monotone DOWNHILL. Closed as a post-hoc filter;
the measurement stands.

### ⭐ The version nobody has tried: TWO PASSES

Every attempt above computes multi-frame features from a GREEDY backward
chain, and that is exactly what the degradation counters say fails. The
ILP already produces a globally optimal capacity-2 solution whose tracks
are connected -- far better trajectories than the greedy pass.

**Solve once, build the multi-frame features from the SOLVED tracks,
re-solve.** That is not circular in the way it first sounds: the
features are not being used to build the trajectories they come from,
they are being used for a second solve. And it directly removes the
diagnosed failure -- three consecutive positions would be available for
most cells rather than a minority.

Cost: two ILP solves per dataset instead of one, so roughly double
`biohub-thr-sweep`'s ~10 min per configuration. Pure CPU on cached
detections. Untried.

## 2026-08-25 -- ⭐⭐⭐ the detection ceiling: 13.1% of cells have NO peak at any threshold, and 9.5% are thrown away after having one

`det_floor.py` -> `biohub-det-floor` v1, **16 min on a T4**, 41 crops,
every 20th frame, 50,897 GT points. Two questions in one pass.

**Anchor holds:** FULL at floor 0.05, gate 0.15 reads F1 0.7554 against
`det_ablate3` v3's 0.7539 (that run used every 10th frame).

### 1. The peak-finding floor is NOT a lever. Closed.

"Recall ceiling" = the fraction of GT points with ANY peak within 5 um,
before NMS and before any gate. Nothing downstream can exceed it.

| config | floor | peaks | ceiling |
|---|---|---|---|
| FULL | 0.05 | 64,584 | **0.8688** |
| FULL | 0.01 | 68,026 | **0.8690** (+0.0002) |
| B0 | 0.05 | 61,144 | 0.8610 |
| B0 | 0.01 | 63,405 | 0.8627 (+0.0017) |
| B2 | 0.05 | 51,282 | 0.7005 |
| B2 | 0.01 | 61,604 | 0.7712 (**+0.0707**) |

**Dropping the floor from 0.05 to 0.01 buys FULL +0.0002.** The F1
curves are IDENTICAL for every gate at or above 0.05 and strictly WORSE
below it -- the extra peaks are junk.

**The cells under 0.05 are not there.** They are not "below the
threshold", the detector produces no response for them at all. Do not
re-open this by lowering the floor further; `cache_train.py`'s 0.05 is
already the right choice and the caches need no rebuild.

### 2. ⭐ The number that matters: where the missing recall actually is

| | share of GT | recoverable by parameters? |
|---|---|---|
| detected today (FULL, gate 0.15) | **0.7738** | -- |
| **has a peak but is discarded** | **0.0950** | **YES** |
| no peak at any threshold | **0.1312** | **NO -- needs a better detector** |

**9.5 percentage points of recall sit in peaks that EXIST and are then
thrown away**, by `_physical_nms` (`NMS_UM` 4.0) or by the gate
(`UNET_THRESH` 0.15). That is five times the size of what
`biohub-thr-sweep` moved by touching the gate, and **`NMS_UM` has never
been swept.**

It points the same way as an unexplained number already on record:
`biohub-div-mass` found **44.7% of divisions had BOTH daughters inside
one watershed label** -- one peak serving two cells. NMS merging nearby
cells would produce exactly that.

⚠ **The ceiling is OPTIMISTIC and by the same mechanism.**
`ceiling_hits` counts a GT point as reachable if any peak lies within
5 um, with **no 1:1 constraint**, so two adjacent cells sharing one peak
are both counted as reachable. The true 1:1 ceiling is lower than
0.8688, and the gap between the two is itself a measure of how much NMS
merges. Worth computing directly.

### 3. B2 settled: genuinely the weakest, now measured rather than assumed

The user objected to `det_ablate3` v2's "B2 is nearly useless" and was
right that it was censored -- v3's sweep bottomed out at the
peak-finding floor with B2's curve still rising. At floor 0.01 the curve
**turns over inside the range**: 0.6712, **0.6744 (gate 0.02)**, 0.6741,
0.6693 ...

**B2's true best F1 is 0.6744**, against B0's 0.7536 and FULL's 0.7554 --
still last by about 0.08. Its recall ceiling also rises +0.0707 when the
floor drops, confirming the calibration diagnosis: B2's probabilities
really are systematically lower. The objection was correct, the
correction is now complete, and the ordering survives it.

## 2026-08-25 -- ⭐⭐⭐ flip-and-average: same F1, an 11% DIFFERENT detection set. The reverse proof falsified my claim.

`tta_probe.py` -> `biohub-tta-probe` v1, **17 min on a T4**. The user
asked for the reverse proof of "the four flip views buy nothing" -- the
claim I made from `biohub-det-ablate3` v2's F1 0.7546 (3 passes) against
0.7539 (12 passes). It is FALSE in the sense that matters.

### A. The four views are correlated but NOT the same volume

| | |
|---|---|
| Pearson r, view 0 vs views 1-3, per branch per frame (n=1845) | median **0.9781** [p10 0.9402, p90 0.9926] |
| spread ACROSS the 4 views / spread across voxels | median **0.1085** [p10 0.061, p90 0.180] |
| mean per-voxel abs difference, heatmap(4) vs heatmap(1) | 0.00198 |

A convolution is translation-equivariant by construction but **nothing
in the architecture makes it flip-equivariant** -- a kernel that responds
to "bright left, dark right" cannot respond identically to the mirror.
Flip invariance is LEARNED from augmentation, so it is approximate and
never exact. The 11% across-view spread is that residual, and it is
exactly what flip-averaging averages away.

This was predicted by `det_ablate3` v3 before the run: flip-averaging
buys about +0.003 F1 for a SINGLE branch (B0 +0.0028, B1 +0.0040, B2
+0.0029), which is impossible if the four views were identical volumes.

### B. The F1 difference is inside its own error bar

| | |
|---|---|
| paired over 41 crops, one view minus four | **+0.00109, SE 0.00150** |
| in units of its own SE | **+0.73** |
| crops where ONE view wins | **19 of 41** |

19 of 41 is a coin flip. **Neither "buys nothing" nor "slightly better"
is supported.** v2 pooled two F1 numbers and reported a 0.0007 gap with
no error estimate; the honest statement was always "indistinguishable at
this sample size".

### ⭐ C. And yet they find DIFFERENT CELLS -- 11% of them

| | |
|---|---|
| paired within 2 um | 50,560 |
| only in the FOUR-view set | 2,786 |
| only in the ONE-view set | 3,512 |
| **symmetric difference** | **6,298 of 56,858 = 11.08%** |

The COUNT gap was 1.3%. The SET gap is **11%**. About one detection in
nine differs. **These are two different detectors that happen to score
the same.** Count agreement is not set agreement, and F1 agreement is
not detector agreement.

### What this changes

**RETRACTED:** "three forward passes tie twelve, so switch and spend the
saved runtime on the lower gate." Switching changes 11% of the
detections, and `biohub-node-filter` measured that detection quality and
adj_J do not move together -- removing detections costs edges faster
than it removes noise. An 11% churn in the node set propagates into
linking, the ILP, and the division patch, and F1 says nothing about
where it lands.

**STANDS:** three passes are not WORSE than twelve on detection F1, and
that is a real 4x saving *if* a pipeline run confirms adj_J holds. It
has to be scored, not assumed. The same caution applies to `B0(4)` and
`B01(8)`, which the ablation listed as ties.

### Still untested

The prediction that the disagreement concentrates on LOW-SCORING,
near-threshold detections -- confident cells should be found by every
view. `set_overlap` counts the symmetric difference but does not record
the scores of the 6,298 detections in it. Cheap to add, and it would say
whether an 11% churn is 11% of the borderline cases or spread across the
whole score range. Those are very different risks.

## 2026-08-25 -- ⭐⭐⭐ the detector ablation, finally run: the ensemble buys DIVERSITY, and 3 passes tie 12

`det_ablate3.py` -> `biohub-det-ablate3` v3, **17 min on a T4**. The
question is the user's from 2026-08-24 and had never been answered; v1
and v2 of the old `det_ablate` never produced a number. Zebrahub dense
GT, greedy 1:1 within 5 um, **every configuration at ITS OWN best gate**.

**Anchor holds:** FULL reads recall 0.7738 / precision 0.7350 against
`biohub-node-bright` v2's 0.7659 / 0.7362 on the same 41 crops (that run
used every 5th frame, this every 10th).

| config | passes | gate* | recall | prec | **F1\*** | F1@0.15 |
|---|---|---|---|---|---|---|
| B0 (`bright`, NO tophat) | 4 | 0.15 | 0.7812 | 0.7235 | 0.7512 | 0.7512 |
| B1 (`traintophat`) | 4 | 0.20 | 0.7936 | 0.6915 | 0.7391 | 0.7251 |
| B2 (`v2_tophat_b32`) | 4 | 0.05 | 0.6699 | 0.6686 | 0.6693 | 0.6239 |
| **B01** | 8 | 0.20 | 0.7700 | 0.7474 | **0.7585** | 0.7522 |
| B02 | 8 | 0.10 | 0.7787 | 0.7177 | 0.7470 | 0.7217 |
| B12 | 8 | 0.10 | 0.8068 | 0.6651 | 0.7291 | 0.7289 |
| FULL | 12 | 0.15 | 0.7738 | 0.7350 | 0.7539 | 0.7539 |
| **FULL_f1** | **3** | 0.15 | 0.7796 | 0.7312 | **0.7546** | 0.7546 |
| FULL_f2 | 6 | 0.15 | 0.7762 | 0.7327 | 0.7539 | 0.7539 |
| **B0_f1** | **1** | 0.15 | 0.7817 | 0.7178 | 0.7484 | 0.7484 |
| B1_f1 | 1 | 0.20 | 0.7958 | 0.6830 | 0.7351 | 0.7193 |
| B2_f1 | 1 | 0.05 | 0.6841 | 0.6495 | 0.6663 | 0.6369 |

### ⭐ The ensemble buys DIVERSITY, not member quality

| pairing | v2 (all at 0.15) | **v3 (own gates)** |
|---|---|---|
| B01 - B12 | +0.0233 | **+0.0294** |
| B02 - B12 | -0.0072 | **+0.0179** |

**At fair gates, every pairing containing B0 beats the pairing without
it** -- and B0 has the WORST reported `val_recall` of the three (0.5731
against 0.6589 and 0.5789). B0 is also the only branch that does not get
top-hat input; branches 1 and 2 are pixel-identical in what they see.
The one genuinely different view of the data is the one the ensemble
cannot do without, and its solo metric said it was the weakest.

**`val_recall` anti-predicts here**, which is now the eighth time this
session an offline number has run opposite to the measured outcome.

### The user caught a real error, and the correction is partial

v2 scored every configuration at 0.15 and read B2's F1 of 0.6239 as
"nearly useless". The user objected that this needed more care. Correct:
**0.15 is calibrated for the THREE-BRANCH AVERAGE, and a single branch
lives on a different probability scale.** At its own gate B2 gains
**+0.045** (0.6239 -> 0.6693).

Two things the correction does NOT overturn, and one it leaves open:

- B2 is still last by ~0.07, far outside noise.
- **B2 does not help the ensemble either: FULL - B01 = -0.0046**, i.e.
  removing B2 from the three is slightly BETTER, and B01 at gate 0.20 is
  the best F1 of all twelve configurations.
- ⚠ **B2's optimum is CENSORED.** Its F1-vs-gate curve falls
  monotonically from the lowest gate swept (0.6693, 0.6585, 0.6499,
  0.6239, 0.5895 ...), so its peak is at or below 0.05 -- and 0.05 is
  `DETECT_THRESH`, the peak-finding floor, where no peaks exist below.
  Measuring B2's true best needs a fresh detection pass with a lower
  floor. Same for `B2_f1`.

### Flip-and-average: helps ONE branch, does nothing on top of three

| | 4 views minus 1 view |
|---|---|
| B0 alone | **+0.0028** |
| B1 alone | **+0.0040** |
| B2 alone | **+0.0029** |
| **FULL (three branches)** | **-0.0007** |

**SUPERSEDES what v2 was read as saying.** Reporting FULL_f1 0.7546
against FULL 0.7539 as "the four flip views buy nothing" was wrong twice
over: the 0.0007 had no error bar, and the flips DO buy about +0.003 --
for a single branch. Averaging across branches already performs the same
variance reduction, so the two are redundant with each other and the
second one is free to drop.

This also predicts what `biohub-tta-probe` must find: **the four views
cannot be identical volumes**, or averaging them could not have moved a
single branch by +0.003.

### ⭐ What is cheap enough to matter

Configurations within 0.003 F1 of FULL, cheapest first:
**`FULL_f1(3)`, `B0(4)`, `FULL_f2(6)`, `B01(8)`, `FULL(12)`.**

**Three forward passes tie twelve.** That is a 4x cut in detection cost
on every future GPU run, and it bears directly on the submission: the
global solve is projected at ~11.8 h against a 12 h cap (ref 55707206's
description), which is what currently makes the +0.0033 threshold change
from `biohub-thr-sweep` too risky to submit on its own.

> **SUPERSEDED 2026-08-25 — that verdict rests on a number nobody
> measured.** 11.8 h is `210 s/dataset × ~200 hidden volumes`, and both
> factors are assumptions: the 210 s was timed on the 4 VISIBLE movies,
> and the ~200 comes from the train split having 199, with nothing
> establishing the hidden split is the same size. Nothing on record
> measures the hidden rescore's duration at all. The user's standing
> observation — no submission has ever still been running the next
> morning — is the only direct evidence and puts it under 8 h. **So
> "too risky to submit on its own" is not established**, and the
> threshold change should be reconsidered on its merits. See
> `kaggle-toolchain.md` section 9 for the measurement that would settle
> it.

> WARNING **QUALIFIED the same day by `biohub-tta-probe` (entry above).**
> "Tie" means tie ON F1. The two configurations disagree on **11.08% of
> their detections** (symmetric difference 6,298 of 56,858, matched
> within 2 um), so switching is not a free saving -- it is a DIFFERENT
> detector with the same summary score, and adj_J does not follow
> detection F1. Score it through the pipeline before spending the
> saving. The same caveat applies to `B0(4)` and `B01(8)`, which this
> entry also lists as ties.

⚠ **This is DETECTION F1, not adj_J.** `biohub-node-filter` measured
that the two do not move together -- removing detections costs edges
faster than it removes noise. A configuration change has to be scored
through the pipeline before it can be believed, and the gate that
maximises F1 is not necessarily the gate that maximises adj_J.

## 2026-08-25 -- ⭐⭐⭐ morphology in the EDGE RANKER: adj_J -0.0015. The whole morphology line is closed.

`morph_edge.py` -> `biohub-morph-edge` v2, **106 min on CPU**, trained on
187 datasets. Six SIGNED morphology columns appended to the ranker's 18:
four child/parent ratios (mean, max, std, MASS) and the two absolute
brightness levels.

**BASE reproduced the anchor exactly** -- score 0.9386, adj_J 0.9254,
div_J 0.1316, TP/FP/FN 5/26/7 -- so the comparison is valid.

| arm | cols | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASE | 18 | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| MORPH | 24 | 0.9326 | 0.9240 | 0.0857 | 3/23/9 |
| delta | | -0.0060 | **-0.0015** | -0.0459 | |

**adj_J -0.0015 is below the ~0.003 holdout floor. Morphology does
nothing for the edge ranker.** Stage A agreed in advance: edge AUC
0.9866 against 0.9864, and permutation importance for all six columns
between +0.0003 and +0.0007.

### The div_J drop is NOT attributable, and saying so matters

div_J fell 0.0459, which unpacks to **TP 5 -> 3: two divisions**. The
standing warning applies -- "⚠ Five true divisions. Every row rests on
TP counts of 1-5, far under anything resolvable alone."

More importantly the pre-registered mechanism is ruled out by this run's
own Stage A. The worry written into the script was that a
morphology-aware ranker would push mother->daughter edges DOWN, refuse
both daughters, and starve the orphan patch (whose proposals need a
parent with exactly one child). That requires the model to SEE the size
change. It cannot: **`MASS_ratio` over 225,861 rows is median 0.994,
p10 0.722, p90 1.270** -- a daughter should sit near 0.5 and the tenth
percentile does not approach it. The script's own pre-registered reading
of that distribution was "if the p10 does not reach anywhere near 0.5
the division signature is not in this feature and the orphan-starvation
worry is moot."

So: **no effect on edges, and a two-event move in divisions that this
sample cannot resolve.** Not "morphology starves divisions".

### The morphology line, closed end to end (2026-08-25)

| route | measurement | verdict |
|---|---|---|
| mass conservation, voxel count | per-fork AUC 0.605 | wrong quantity measured |
| fringe vs tessellation | 76.8% background-bounded, density slope -0.028 | **user right: fringe** |
| mass conservation, brightness | per-fork AUC 0.621, +0.0157 | fringe real, nothing under it |
| box brightness as a node filter | F10 - S10 adj_J **-0.0297** | loses to raising the threshold |
| morphology in the edge ranker | adj_J **-0.0015** | no effect |

Underneath all five: **no dataset here annotates cell size**, so size is
unverifiable rather than merely weak (see the ruler entry). Brightness
escapes that -- its label exists -- and brightness genuinely separates
real detections from hallucinations at AUC 0.866 on Zebrahub. It just
does not survive the transfer to competition data, and removing
detections is expensive under an edge-counting metric regardless.

**Do not reopen this with a better mask algorithm or another feature
built on masks.** Reopen it only if a size or segmentation annotation
appears, or if the filter can be trained on competition data, which
needs a spurious-detection label that sparse GT cannot give.

## 2026-08-25 -- ⭐⭐⭐ the node filter LOSES to raising the threshold. Cutting detections is expensive.

`node_filter.py` -> `biohub-node-filter` v1, **66 min on CPU**. Assembled
from `poolsize.py` so environment, caches, holdout and ILP are identical
to what produced the anchor; the D187 ranker is LOADED from
`biohub-poolsize`'s pkl rather than retrained, so the linker is
submission 55757064's byte for byte.

**BASE reproduced the anchor exactly** -- score 0.9386, adj_J 0.9254,
div_J 0.1316, TP/FP/FN 5/26/7 -- so every comparison below is valid.

| arm | cuts | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| **BASE** | nothing | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| F10 | bottom 10% by node filter | 0.8879 | 0.8736 | **0.1429** | 5/23/7 |
| S10 | bottom 10% by det score | 0.9168 | 0.9033 | 0.1351 | 5/25/7 |
| F20 | bottom 20% by node filter | 0.8150 | 0.8029 | 0.1212 | 4/21/8 |
| S20 | bottom 20% by det score | 0.8878 | 0.8753 | 0.1250 | 4/20/8 |

**F10 - S10 = -0.0297 adj_J. F20 - S20 = -0.0724.** At a matched cut
count the learned filter is WORSE than simply raising the detection
threshold, despite leading it by +0.1418 AUC offline on Zebrahub.

### Two findings, and the second is the more useful one

**1. The Zebrahub -> competition transfer failed.** The run was designed
against exactly this risk -- cutting by PERCENTILE within each dataset so
only the ORDERING had to survive, never the probability scale. The
ordering did not survive either. `morph_cheap`'s rho 0.965 licenses box
statistics as a stand-in for mask statistics WITHIN a domain; it says
nothing about ranking detections across domains, and this run is the
evidence that it does not carry.

**This is the FIFTH time an offline gain reversed in the pipeline** --
after scale-invariant edge features, the division model's four arms,
D187's own edge AUC, and div-gate's prune.

**2. Removing detections is expensive under adj_J, by a mechanism that
generalises.** Cutting 10% by the BEST EXISTING SIGNAL, the detection
score itself, already costs **-0.0221 adj_J**; 20% costs -0.0501. The
metric counts EDGES: a real cell carries roughly two of them and loses
both when it is cut, while a false detection links to little and so
returns little when removed. Any node filter must therefore be far more
precise than the detection score merely to break even, and none of these
was.

The one signal pointing the right way: **F10 raised div_J to 0.1429 and
cut false divisions 26 -> 23** with TP and FN unchanged, i.e. it removed
hallucinations that were feeding spurious forks. div_J carries a tenth
of adj_J's weight, so -0.0518 adj_J buries +0.0113 div_J.

### What would have to change for this line to live

Train the filter on COMPETITION data. That needs a "this detection is
spurious" label, which sparse GT cannot give -- an unmatched detection
there means unannotated, not false. That absence is why Zebrahub was
used, and the transfer failure is why that substitution did not work.
**Both ends of this line are blocked by the same missing label.**

### ⚠ An open question this run RAISES, with the evidence conflicting

If removing detections is expensive because missing cells cost edges,
the asymmetry may run the other way: `UNET_THRESH` is 0.15, and LOWERING
it would add detections, buying recall at the price of precision.

Conflicting evidence, both already on record, neither decisive:

- FOR: this run measures a controlled intervention and it says removing
  real cells is what hurts.
- AGAINST: `corr(edge J, N/N_true) = -0.64` says over-detection tracks
  worse edge J -- but that is a correlation ACROSS datasets, confounded
  by density, not an intervention.

A threshold sweep would settle it and is the same shape of run as this
one. Nobody has run it downward.

## 2026-08-25 -- ⭐⭐⭐ THE RULER IS THE PROBLEM. Mask-based mass is unmeasurable here, and the line closes.

**User's objection, and it ends the line:** the detector does not even
achieve full recall on Zebrahub, so expecting its masks to be accurate is
asking too much. Checked against the record rather than argued:
**recall 0.65 / precision 0.64** at the pipeline's det gate
(`progress.md` "the clean ruler's first reading: recall 0.65, not 0.99").

**35% of cells have no detection at all and 36% of detections are
spurious.** The watershed's seeds ARE those peaks, so:

- a MISSED cell leaves its volume unclaimed and a neighbouring basin
  swallows it -- that neighbour's voxel count and brightness now contain
  a whole extra cell;
- a SPURIOUS peak splits one real cell into two basins.

And the bias runs in exactly the direction that manufactures 1.713: at t
there is one seed, at t+1 there are two, and the pair can claim not only
the mother's old territory but any of the 35% unclaimed ground nearby.
**The 40% merged-label rate is direct evidence of the same failure** --
the detector found ONE peak where there were TWO cells. An instrument
that cannot count cells cannot measure their size.

### The validation that was not one

`morph_probe` established our `regionprops` reproduces HOCT's
`create_graph` at **0.000e+00** and this was read as "the extractor is
correct". It is not: **it proves two implementations compute the SAME
quantity, not that the quantity is a cell.** Two correct implementations
of the wrong thing agree perfectly.

Same defect as the diameter anchor earlier today (7.08 / 8.83 um also
came from watershed-on-heatmap masks). **A check drawn from the same
construction as the thing under test validates nothing.** Both instances
were caught by the user, both within one day, and this is the general
form: an anchor must come from OUTSIDE the construction, or it is only
measuring self-consistency.

### Why a clean subsample does not rescue it either

Zebrahub's GT is dense, so one could keep only divisions whose whole
local neighbourhood is perfectly detected -- every GT node matched, no
unmatched detection nearby -- roughly 0.65^5 ~ 12% of events, a few
hundred. That could answer the SCIENCE (is mass conserved) but not the
ENGINEERING: **at inference there is no GT, so the pipeline cannot tell
a clean neighbourhood from a dirty one**, and in the dirty majority the
quantity is noise. A feature that is only trustworthy when you already
know the answer is not a feature.

### Status

CLOSED: mass conservation as a division test, on masks built from this
detector. Not because the physics is wrong -- matter does not appear
from nowhere and the user is right about that -- but because **no ruler
here can measure it.** Do not retry with another threshold, another
weighting, or another mask floor; all of them are downstream of the same
seeds.

**This raises the value of detector work.** Poor detection was costed at
~1.75% of the score through missed edge endpoints. It is now also the
thing that makes EVERY mask-based morphology feature untrustworthy --
size, shape, brightness, all of them.


## 2026-08-25 -- ⭐⭐ brightness strips the fringe and the signal is still not there. Size is DONE for divisions.

`div_intmass.py` -> `biohub-div-intmass` v2, **4 min on CPU**, 12 crops
carrying 5,699 division events. Sum BRIGHTNESS inside each mask instead
of counting voxels, so a dim fringe is discounted continuously with no
new threshold -- the retry `mask_anatomy` called for.

**Answer: per-fork AUC 0.6210 by brightness against 0.6054 by voxel
count. +0.0157. Mass conservation does not come back.**

### The gate passed -- brightness really does discount the fringe

| the outer one-voxel shell holds | |
|---|---|
| **volume** | 47.7% |
| **brightness** | **33.3%** |

A 14.4-point discount. The fringe is real and it is dim, exactly as
`mask_anatomy` said. So step 2's numbers are attributable.

### The anchor held to 0.0003

| | div_mass, 41 crops | here, 12 crops |
|---|---|---|
| voxel-count DIV median | 1.772 | **1.784** |
| voxel-count per-fork AUC | 0.6051 | **0.6054** |

The subset reproduces the whole, so the brightness-minus-voxel
difference is the only thing that varied.

### The measurement

| | voxel count | **brightness** | conserved predicts |
|---|---|---|---|
| DIV `(Ia+Ib)/Ip` | 1.784 | **1.713** | 1.0 |
| CONT | 1.000 | **1.000** | 1.0 |
| FAKE FORK | 2.009 | 2.027 | 2.0 |
| **per-fork AUC** | 0.6054 | **0.6210** | |

Stripping the fringe moved the division ratio 4% of the way toward 1.0
when it needed to move about 80%. **The fringe was real but it was not
what was hiding the signal; there was no signal under it.**

### ⭐ The user's photobleaching worry, answered

The user raised that mother-at-t and daughters-at-t+1 are two separate
exposures, so bleaching, focus drift or illumination drift would bias
I(t+1)/I(t) for reasons that are not biology. **`CONT Inext/Inow` is
exactly that control and it reads 1.000 over 45,657 non-dividing
cells.** No drift. (`_norm` also normalises each frame by its own 50th
and 99.5th percentiles, so global drift is removed before this is even
measured -- and the CONT row confirms the removal worked.)

### Two live explanations for 1.713, neither of which changes the call

1. the two daughters genuinely carry ~1.7x the mother's fluorescence --
   either not conserved, or the mother's last annotated frame is not the
   frame where her material is all present;
2. the mother is mid-mitosis at t, the detector responds weakly to
   condensed chromatin, her mask is small and her brightness undercounted.

Also a hard ceiling regardless: **2,289 of 5,699 divisions (40%) had both
daughters in one label** and were dropped, so the 2,870 usable ones are
the already-well-separated ones -- a biased, easier sample, and the
answer is still negative on it.

### ⚠ Where this leaves divisions, checked against the record rather than recalled

I recommended wiring div-kinematics' FULL model (1.8 FP per TP vs BASE's
6.9) into the pipeline. **That was already spent and it failed:**
`biohub-div-gate` did exactly that on 2026-08-24 and the prune sweep is
monotone DOWNHILL -- doing nothing wins, because the bottom 25% of forks
by `p_div` holds 20% of the TPs against only 4% of the FPs. The 1.8 was
measured over every candidate triple `build()` emits; the pipeline emits
553 forks that already passed the ILP or the orphan patch, and those are
the hard cases. Full entry below.

**The correction itself is the reusable part: I produced 1.8 from recall
and did not check whether it had been spent. `working-agreement.md` says
look numbers up at the moment of use -- that rule covers not just the
VALUE but whether the lead is still open.**

The division ledger as it now stands:

| route | status |
|---|---|
| morphology / size as a division signal | measured today, AUC 0.621, too weak |
| `p_div` as a post-hoc fork filter | dead 2026-08-24, monotone downhill |
| forking harder to buy true divisions | dead, division FN never below 6 |
| `div_forks`' three uses | all null 2026-08-24 |
| **morphology's effect on the ORPHAN SUPPLY** | **open -- `morph_edge`'s div_J** |


## 2026-08-25 -- ⭐⭐⭐ the masks are cells plus a FRINGE, not a tessellation. The user was right.

`mask_anatomy.py` -> `biohub-mask-anatomy` v1, **19 s on CPU**, labels
only. Two tests that share no arithmetic; both land on FRINGE.

**This overturns my reading of `div_mass` the same day I wrote it.** I
said a voxel count measures the TERRITORY between peaks and filed the
line closed. The user said it sounds like the edges are being
over-counted, then proposed the method that settled it: assume the
fringe hypothesis, derive what it forces, and check that.

### Test 1 -- when a mask stops, why does it stop?

A fringe ends where the heatmap falls below `MASK_FLOOR`, i.e. at
BACKGROUND. A territory boundary is a watershed line drawn halfway to a
neighbouring peak: cell on both sides, no background anywhere near. So
count boundary faces and ask what is on the other side.

| boundary faces against | count | share |
|---|---|---|
| **BACKGROUND** | 14,911,652 | **76.8%** |
| another mask | 4,508,503 | 23.2% |

Three quarters of every mask surface is cut by the threshold, not by a
neighbour.

### Test 2 -- does a mask shrink when the field is crowded?

The 41 crops span **45 to 809 cells per frame, an 18x range**.

| crop | cells/frame | median vol | bg face % |
|---|---|---|---|
| ZSNS004_z2y5x5 | 45 | 523 um^3 | 95.7% |
| ZSNS005_z1y1x5 | 55 | 451 um^3 | 97.5% |
| ZSNS001_z2y2x7 | 450 | 541 um^3 | 74.3% |
| ZSNS004_z2y1x3 | 606 | 571 um^3 | 81.4% |
| ZSNS005_z1y6x3 | 809 | 534 um^3 | 72.1% |

**log(volume) on log(cells per frame): slope -0.028, r -0.127.**
Territory requires -1 (a cell's share is the volume divided by the
cells). Volume spans 2.4x against density's 18x. A tessellation cannot
do this; a cell plus a fringe can, because neither knows how crowded it
is. Even the densest crop is still 72% background-bounded -- territory
exists but never dominates.

### What this means

`div_mass`'s 1.772 is not evidence that mass is not conserved. It is
evidence that **we weighed the wrong thing**: mask = core M + fringe H
with H = 3.39 M, so ~77% of the volume is a shell about 1.3 voxels thick
on a 6.62-voxel mask. Mass conservation has still never been tested.

**The retry is `div_intmass.py` -> `biohub-div-intmass`: sum BRIGHTNESS
inside the mask instead of counting voxels.** A fringe voxel sits at
heatmap 0.05-0.1 and reads dim; a core voxel is near 0.9 and reads
bright, so brightness discounts the fringe continuously with no new
threshold -- which matters because at 6.6 voxels across there is no room
to erode. It is gated on measuring that discount rather than assuming
it: the outer shell's share of VOLUME against its share of LIGHT, with
step 2 suppressed if they are equal.

### Method note worth keeping

The user's proof-by-contradiction beat the correlation study I was about
to write: **19 seconds against a multi-crop regression, because assuming
a hypothesis and hunting its forced consequence finds a one-number test.**
Also on record: my diameter anchor (7.08 / 8.83 um) could not arbitrate
this at all, because those numbers came from watershed-on-heatmap masks
too and carry the same fringe. An anchor drawn from the same
construction as the thing under test proves nothing about it.


## 2026-08-25 -- ⭐⭐ mass conservation: a voxel count cannot test it. WHY is open.

`div_mass.py` -> `biohub-div-mass` v1, **3 min on CPU**. Reads only
`biohub-zh-masks`; no detector, no images, no competition mount.

**Answer: (Va+Vb)/Vp is 1.772 where mass conservation predicts 1.0, and
a fake fork is 1.984 where it predicts 2.0. The two are nearly the same
number, so a voxel count on these masks cannot tell a real division from
a mother plus a passing neighbour. Per-fork AUC 0.605.**

> ⚠ **CORRECTED same day, and the correction matters.** This entry
> first said the cause was settled -- that the watershed measures
> TERRITORY between peaks rather than cell mass -- and filed the line
> CLOSED. That was over-claimed. The user proposed the competing
> reading: the masks are inflated by a fuzzy FRINGE and the real signal
> is buried under it. Writing mask = core M + fringe H, a division reads
> (M+2H)/(M+H), which equals the measured 1.772 at **H = 3.39 M** -- the
> cell would be 22.8% of the mask, a shell only ~1.3 voxels thick on a
> 6.62-voxel mask, since volume goes as the cube. A fake fork reads 2.0
> and a continuation 1.0 under EITHER reading. **Every number below fits
> both, and they prescribe opposite next moves:** territory means no
> threshold helps, fringe means strip it and the signal returns.
> `mask_anatomy.py` -> `biohub-mask-anatomy` decides it. Until that
> lands, the measured failure stands and the CAUSE does not.
>
> Note also that the diameter anchor cannot arbitrate: 7.08 / 8.83 um on
> the competition data came from watershed-on-heatmap masks too, so both
> could be inflated by the same fringe.

### Both gates and both anchors passed, so the negative is attributable

- 974,613 of 1,021,008 GT nodes (**95.5%**) land inside a label.
- median equivalent diameter **10.76 um** [9.00, 13.54]. Anchors: 7.08 um
  over all competition detections, 8.83 um over GT-matched ones. Zebrahub
  cells slightly larger; same order, so the pooled indexing is right.
- mining: **7,894 division events, 0 crops empty** (15,788 before the
  daughter-order dedupe). The broken-mining signature -- 68 positives,
  26 of 41 crops empty -- did not appear. 945,261 continuations.

### The measurement

| quantity | mass conservation predicts | measured |
|---|---|---|
| DIV `(Va+Vb)/Vp` | **1.0** | **1.772** [1.03, 2.93] |
| DIV `Vchild/Vmother` | 0.5 | 0.878 [0.43, 1.53] |
| DIV diameter ratio | 0.794 | 0.958 [0.76, 1.15] |
| CONT `Vnext/Vnow` | 1.0 | **1.000** [0.73, 1.37] |
| FAKE FORK `(V1+Vq)/V0` | 2.0 | **1.984** [1.31, 3.18] |

per-FORK AUC (true division vs fake fork) **0.6051**; per-EDGE AUC
(daughter vs continuation) **0.6247**.

### Why this is a clean negative and not a broken run

**The two control rows hit their predictions exactly.** Continuations
read 1.000 and fake forks read 1.984. The machinery reproduces the
physics wherever the physics is about COUNTING CELLS. It fails only on
the row where the answer depends on cell SIZE.

One explanation covers every number: the mother is ONE peak and so ONE
basin; the two daughters are TWO peaks and so TWO basins; basins are
sized by neighbour spacing, not by the cell. The ratio is then ~2
regardless of whether mass is conserved -- which is why true divisions
(1.772) and fake forks (1.984) are nearly the same number, and why the
AUC is 0.60 rather than 0.5 or 0.9.

Two corroborations:

- **CONT diameter ratio is 1.000 with p10-p90 of [0.900, 1.111].** Real
  cell volume does not hold to +-10% frame to frame. That tightness is a
  tessellation, not a measurement of matter.
- **44.7% of divisions had BOTH daughters inside ONE label** (3,532 of
  7,894; a further 678 lost a member to non-detection, leaving 3,684
  usable). Freshly separated sisters touch, and a watershed seeded on a
  detection heatmap does not split them. That fraction is a hard ceiling
  on what ANY size feature could see here.

### What this does and does not close

MEASURED and not in dispute: a VOXEL COUNT on these masks separates true
divisions from fake forks at AUC 0.605. As a feature, that is nothing.

OPEN: **why**. See the correction at the top. `mask_anatomy.py` asks the
one question that splits the two readings using nothing but the label
array -- what does a mask's surface actually touch? A fringe ends where
the heatmap falls below `MASK_FLOOR`, i.e. at BACKGROUND; a territory
boundary is a watershed line halfway to a neighbour, with cell on both
sides. It cross-checks with density: across the 41 crops, 44 to 815
cells per frame, territory predicts median mask volume falling as 1/N
(log-log slope near -1) and fringe predicts flat (near 0).

NEVER MEASURED: whether cell mass is actually conserved at division. An
intensity-weighted mass (background-subtracted brightness sum inside the
mask, which `morph_edge.py`'s `crop_stats_all` already computes as its
4th column) is a property of the cell rather than of the region, and
discounts a dim fringe for free. It needs the Zebrahub images mounted
alongside the masks, still CPU. **Weigh it against the 44.7% ceiling
before spending a slot** -- that fraction is a detector failure (one
peak served two daughters) and neither reading above touches it.


## 2026-08-25 -- masks for all 41 Zebrahub crops: the morphology line is now free

`zh_masks.py` -> `biohub-zh-masks` v1, **54 min GPU, 41/41 written, 0
skipped, no degradation warning fired.** Watershed on the detector
heatmap, pooled to an isotropic 1.625 um grid, one `.npz` per crop
carrying `labels` plus the GT (`nodes`, `track_id`, `parent_track_id`,
`voxel_um`, `t0`), so a consumer needs this one kernel and not the 7.5 GB
crops.

T=100 for every crop. Peaks per crop 4,438 (`ZSNS004_z2y5x5`) to 81,493
(`ZSNS005_z1y6x3`); 0.5-5.7 MB each.

**Why this run existed:** `regionprops` on these masks IS HOCT's
`create_graph`, verified at 0.000e+00 (entry below). One GPU pass buys
unlimited CPU iteration on size and shape -- which is what the four
wasted HOCT GPU runs were paying for.

## 2026-08-25 -- morphology change is a REAL edge signal, not detection quality leaking

`morph_confound.py` -> `biohub-morph-confound`, CPU. The worry was that
"morphology separates true from false links" is really "morphology
separates real cells from spurious detections", since a false link's far
endpoint is usually a spurious detection. Test: split the negatives by
whether the far endpoint is itself a GT-matched cell, on Zebrahub's
DENSE ground truth so real-endpoint negatives actually exist.

| negative class | n | AUC of \|d intensity\| |
|---|---|---|
| **N-real** (both endpoints real cells, simply not linked) | 2,022,702 | **0.8298** |
| N-fake (far endpoint spurious) | 88,941 | 0.8640 |

**0.8298 on two million negatives whose both ends are real cells.** The
detection component exists and is worth 0.034. The signal is not it.

This is what the competition data could not answer: its GT is sparse, so
"a matched cell near another matched cell" is nearly always its own
successor -- the same construction that produced `morph_probe` step 2b's
3 negatives and every HOCT gate's "109 of 109".

### `morph_cheap`: which of these can be had without masks

`morph_cheap.py` -> `biohub-morph-cheap`, CPU, Spearman against the
verified mask-based `regionprops` (ranking is all a tree needs):

| mask feature | best box proxy | rho | usable |
|---|---|---|---|
| `intensity_mean` | crop mean | **0.965** | yes |
| `intensity_max` | crop max | **0.968** | yes |
| `intensity_std` | crop std | **0.946** | yes |
| `equivalent_diameter_area` | `n_half` | 0.713 | **NO** |

**A box does not know where the cell ends.** Every size proxy failed.
So intensity is free from the raw volume; SIZE needs the masks, which is
why `zh_masks` was worth 54 minutes of GPU.


## 2026-08-25 -- ⭐⭐ morphology is OURS, and real cells are 28% bigger than spurious ones

`morph_probe.py` -> `biohub-morph-probe` v2, 6 min on CPU. The user
objected to testing morphology the fast way -- computing it a NEW way
while asking a NEW question means a negative result cannot be attributed
-- so the run is gated: step 2 is SUPPRESSED unless step 1 proves our
extractor reproduces HOCT's node for node.

### Step 1: MATCH, and v1's mismatch was worth having

v2: all five columns agree at **0.000e+00** over 770/770 nodes paired by
centroid. **`skimage.regionprops` on the cached masks IS `create_graph`.**
hoct can be dropped entirely -- no 25 MB weights, no offline packaging.

v1 MISMATCHED, and the shape of it named the cause in one step:
`equivalent_diameter_area` agreed to 0.000e+00 while all four intensity
columns differed (worst 9.4e-01). Same masks, same 770 nodes, so it had
to be the intensity IMAGE, not the regions. It was: `create_graph`
defaults to `normalize_images=True` and we passed images `_norm` had
already normalised, so HOCT normalised twice. Ours read i_mean median
0.6807 against its 0.4948.

**That is the fourth inherited default this session that nobody had
checked** -- after "the metric needs a GPU" (really `load_image=True`),
the ranker's 18 transplanted features, and its 32-dataset pool. Without
the gate we would have carried ~40% wrong intensities into the
hypothesis and been unable to attribute the answer.

### Step 2a: the over-detection problem finally has a signal

127 of 770 detections matched GT; the other 643 are what the node-count
penalty pays for. Medians, with [p10, p90]:

| | matched | unmatched | ratio |
|---|---|---|---|
| `diam` | **8.83** [7.08, 10.22] | **6.87** [5.35, 8.02] | **1.28** |
| `i_mean` | **0.856** [0.413, 1.117] | **0.654** [0.305, 0.979] | **1.31** |
| `i_max` | 1.405 | 1.266 | 1.11 |
| `i_std` | 0.216 | 0.233 | 0.92 |

**Real cells are 28% bigger and 31% brighter than spurious detections.**
The distributions overlap -- this is not a clean cut -- but `N/N_true`
running to 1.30 has been on record for days with no tool able to touch
it, because every feature this project owns is a coordinate or a
detection score. Size and brightness are a new channel.

### Step 2b: UNUSABLE, my design error, same one as the HOCT gates

114 candidate pairs, 111 true -- **3 negatives**. The printed ratios (0.58
and 0.38) are computed on three samples and mean nothing.

Cause: the pairing required BOTH endpoints to be GT-matched. GT is
sparsely annotated, so within 15 um of a matched cell the only other
matched cell is nearly always its own successor. The negative class is
filtered out by construction -- **exactly what produced the "109 of 109"
in every HOCT gate**. Fix: draw negatives as "matched parent + any
candidate that is not its GT successor", without requiring the candidate
to be matched.

⚠ **1 division in this dataset's 30 frames.** "Does the mother shrink
before dividing" needs orders of magnitude more; size that run against
this number.

## 2026-08-25 -- HOCT: gate CLOSED, negative. Its FEATURE EXTRACTOR is the keeper.

`hoct_score.py` -> `biohub-hoct-score` v4, 7 min on CPU. Candidate graph
with scores, 364 edges whose endpoints both matched GT, 111 real
(30.5% positive) -- so unlike the four earlier attempts there are
negatives and the AUC means something.

| | AUC |
|---|---|
| HOCT `similarity`, blind (v2, no images) | 0.6673 |
| HOCT `similarity`, with images (v4) | **0.7012** |
| **negative distance, same edges** | **0.9091** |

**It loses to "closer is better" by 0.21.** Images helped 0.034 and did
not come close to closing it. There is no fine-tuning path -- the
training code is withheld -- so the line is closed.

The domain gap was the stated risk and it landed: `general_v0` was
trained on Cell Tracking Challenge plus a bacteria benchmark, not
light-sheet embryos. Bounds worth keeping: one dataset, 30 frames, and
our masks are WATERSHED ON A DETECTION HEATMAP, not real segmentation --
Bragantini's own caveat is that results depend on mask quality. Better
than the rank-24 competitor's uniform balls, still synthetic.

### ⭐ What to keep: create_graph is a morphology extractor

`create_graph` computes per-node features from the masks, on CPU, in
under a second per frame. All live and non-degenerate on our data:

| column | median | p10 | p90 |
|---|---|---|---|
| `equivalent_diameter_area` | 7.08 | 5.41 | 8.95 |
| `intensity_mean` | 0.4948 | 0.2347 | 0.7242 |
| `border_dist` | 0.4812 | 0.0000 | 0.6102 |
| `orphan_prob` | 0.0775 | 0.0455 | 0.2609 |

plus `inertia_tensor` and `intensity_min/max/std`.

**Every feature this project owns is a coordinate or a detection score.**
Cell SIZE, SHAPE and BRIGHTNESS have never been available to any model
here. `division_features.py` was written to ask whether they separate a
dividing cell and has no recorded result anywhere.

**This reopens what the division entry above parked.** That entry says
the remaining division routes "need pixels" and are therefore expensive,
because reading morphology meant re-reading raw zarr per candidate. It is
a CPU function call over masks we already know how to build -- the same
watershed, on the pooled heatmap, which is isotropic for free. The
"too expensive" filing was wrong and should be revisited before the
division line is called finished.

⚠ The masks cost one GPU pass and `biohub-hoct-gate` v4 has them cached
(123 KB npz). Everything downstream is free.

### ⚠ SUPERSEDED 2026-08-25 by the morphology line's close (user's question)

**The table above is not independent evidence that morphology is
measurable here.** `create_graph` computes its features FROM THE MASKS,
and the masks are ours -- watershed on a detection heatmap, as this same
entry says two paragraphs up. The 7.08 um came from the same
construction `biohub-mask-anatomy` later showed is 76.8%
background-bounded, i.e. mostly fringe. HOCT ran its code over our
numbers; it did not supply an outside measurement.

The reason the pretrained model can rely on morphology at all is that
`general_v0` was trained on Cell Tracking Challenge data, **which has
real segmentation ground truth**. We have none: Zebrahub's GT is points
and the competition's is tracks. That asymmetry is the whole story of
the morphology line, and it was knowable here.

**The warning was already in this entry and went unacted on:**
"Bragantini's own caveat is that results depend on mask quality."

The split it implies was also derivable in advance, and matches what
five runs on 2026-08-25 measured:

| `create_graph` column family | depends on the mask boundary | outcome |
|---|---|---|
| `equivalent_diameter_area`, `inertia_tensor` | entirely | DEAD -- unverifiable, no size annotation exists |
| `intensity_mean/min/max/std` | weakly | ALIVE -- AUC 0.866 separating hallucinated detections, but does not transfer Zebrahub -> competition |

**Rule this yields: a feature is only as trustworthy as the most
derived thing it depends on.** Before adopting a feature, ask what
annotation would falsify it. If none exists, it cannot be validated no
matter how good it looks.

## 2026-08-25 -- THE RANKER TRAINS ON 32 DATASETS. 187 ARE CACHED. +0.0073 LOCALLY, NOTHING ON THE LB.

> SUPERSEDED 2026-08-25 by the leaderboard. Submission 55757064
> scored **0.920** against the D32 baseline's 0.921, with hidden-test
> output within 0.07%. The +0.0073 below is real on the 12 `6bba`
> holdout datasets and absent on the hidden test. The entry stands as
> a record of what was measured; it is not a description of an
> improvement. See the LB entry at the top.

`poolsize.py` -> `biohub-poolsize`, 39 min on **CPU**. Both arms use the
same 18 columns, the same code, the same protocol and the same holdout.
The ONLY difference is how many datasets the edge ranker saw. Rows are
built once over all 187 and each arm selects its subset by which mount
the cache came from, so D32 is `ranker_fine`'s pool exactly, not a random
32.

| arm | datasets | pairs | edge AUC | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|---|---|
| D32 | 32 | 39,792 | **0.9884** | 0.9313 | 0.9222 | 0.0909 | 3/21/9 |
| **D187** | **187** | **225,861** | 0.9864 | **0.9386** | **0.9254** | **0.1316** | **5/26/7** |

**adj_J +0.0032, score +0.0073.** Both anchors hold: D32's edge AUC is
**0.9884**, the exact figure `ranker_fine` recorded, and D32's adj_J
0.9222 sits 0.0002 from div-gate's 0.9220 with the saved pkl. So the
delta is the pool and nothing else.

### Where the unused data came from

`ranker_fine` learned on `biohub-edge-ranker`'s 44 caches minus the 12
holdout = 32. `biohub-cache-train` cached the other 155 on 2026-08-23 for
the DIVISION census, and its own header lists "the edge ranker's training
pool goes from 32 datasets to 187" as payoff #2. **Nobody ever spent it.**
Every ranker since has loaded the 32-dataset pkl.

**This is free. No new idea, no new dependency, no pixels.** It needs the
submission kernel to train the ranker on all 187 (or to load a pkl that
was), and biohub-cpu-probe2 already proved that whole line runs on CPU.

### Edge AUC anti-predicted the pipeline for the THIRD time today

D187's edge AUC is LOWER (0.9864 vs 0.9884) while its adj_J is higher.
Same inversion as scale-invariant features and as the division model's
four arms. **Stop reading edge AUC.** (Its own drop is partly mechanical
-- a bigger pool means a bigger, harder validation split -- but that is
exactly why it cannot be compared across configurations.)

### A worry that did not materialise

div_J ALSO improved, 0.0909 -> 0.1316, divTP 3 -> 5. The `ranker_fine`
header warned that a more confident 1:1 linker "steals second daughters
into wrong continuations, draining the orphan pool the division patch
mines". A ranker that is more CORRECT, rather than merely more
confident, feeds that mechanism instead of starving it.

⚠ The run ERRORed at the very end on two lines that still said
`res["ACC"]`, arm names left over from `accel.py` -- **after** printing
every number. Fixed. A summary line is not worth a 39-minute run.

## 2026-08-24 -- ⭐⭐⭐ div-forks: the division line is CLOSED. Three uses, all null.

`div_forks.py` -> `biohub-div-forks`, 38 min. div-gate showed the model
could not AUDIT forks; the objection was that pruning was the wrong shape
of intervention, since the model was measured on SELECTION over the full
candidate space. This kernel gave it selection. It still fails.

All three arms start from the same pre-patch graph and the same quotas.

| arm | forks added | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASELINE nearest-first | 442 | 0.9363 | 0.9220 | 0.1429 | 5/23/7 |
| RERANK by `p_div` | 442 | 0.9369 | 0.9222 | 0.1471 | 5/**22**/7 |
| GENERATE, filters off | 595 | 0.9342 | 0.9226 | 0.1163 | 5/**31**/7 |

BASELINE reproduces div-gate's prune-0% row exactly. RERANK buys **one**
fewer false division, +0.0006, an order of magnitude under the resolution
floor. GENERATE is worse. **TP is 5 in all three** -- no arm found a
division another missed.

Per fork added the FP rate is identical: 23/442, 22/442, 31/595 = 5.2%,
5.0%, 5.2%. The model is not separating anything.

### Why -- and it closes the whole thread

The filter stack cuts **14,159 loose proposals to 769**, an 18x
reduction. The quota then takes **442 of those 769**. So the RANKING step
chooses 442 from 769: a 1.7x selection with almost no leverage, and it
cannot decline to fill the quota no matter how bad the remaining
candidates are.

And the filters already contain the model's best feature in crude form:
`DIV_DIVERGE_UM >= 2.25 um` is one-frame divergence, the hand-set ancestor
of `sep_slope` (true 1.155 vs random 0.082 um/frame). **The 18x cut is
done by a rough version of exactly what the model learned, so by the time
the model is consulted the discriminating work is already finished.**

### CORRECTION, same day, after the user pushed back

The first draft of this entry said the division line was exhausted and
that the ranking step had "almost no leverage". **That was wrong, and the
arithmetic says so.** RERANK chooses 442 of 769, so it may EXCLUDE 327.
Only 23 forks are counted FPs. If those 23 sit inside the 769 and can go
without costing a TP, div_J runs 5/35 = 0.143 -> 5/12 = 0.417, worth
**+0.027**. The model removed **one of the 23**.

That is not an absent lever. That is a lever the model failed to pull.
(Discount it for replacements drawn from the remaining 327 being counted
FPs themselves -- the true ceiling is lower than +0.027, but nowhere near
zero.)

So the honest status is **the division MODEL is not good enough**, not
"divisions are unreachable". Two candidate causes, neither tested:

1. **Features.** What the model has that the filter stack does not is
   thin: the filters already encode one-frame divergence, which is the
   crude ancestor of `sep_slope`.
2. **Training data -- and this one is quantitative.** `biohub-zh-divisions`
   validated **468,617** Zebrahub divisions. div-train and div-kinematics
   train on **2001**, because only 41 crops were ever extracted
   (`zebrahub_extract.py`, 7.5 GB) and ~51 of each crop's divisions land
   on detections. `PER_CROP = 400` is NOT binding; the crop COUNT is.
   **We are training on 0.43% of the divisions we have already
   validated.**

Extracting more crops is a known, already-written path. Nobody has
measured whether the division model's AP scales with it.

What IS still established: the funnel's FN 7 (2 detection, 3 gate, 2
floor) and divTP never leaving 5 across every configuration tried.

⚠ Same caveat as every division row: five true divisions on the holdout.

## 2026-08-24 -- ⭐⭐⭐ the edge ranker never sees `edge_prob`, and "window" means two opposite things

Found while answering the user's question "doesn't linking already use
multiple frames -- didn't the edge detector use a window?".

### Two `window_size`s, opposite meanings. Do not confuse them again.

| where | what it is | status |
|---|---|---|
| `window_size = 2`, the official edge head | how many frames the MODEL SEES = two, the minimum. `competition-brief.md:179`: it "structurally **cannot use velocity**" | architectural, not a knob |
| `window_size = 25`, `solve_edges` | ILP time-chunking for SPEED | REVERTED; the global solve is what runs |

Multi-frame evidence has never entered linking. The reverted windowing was
about how the ILP is chunked, nothing more.

### ⭐ The transplant dropped the first-listed feature

`public-solutions-recon.md:300` describes yusuketogashi's ranker as "~25
engineered features -- **primary edge_prob**, raw vs motion-extrapolated
distance and their difference...". Ours has 18 and **`edge_prob` is not
among them**. Its `ps[i]` / `cs[j]` are DETECTION scores from `seeds_of`,
not the edge head's opinion of the edge.

The cause is which cache the ranker reads:

| cache | keys | read by |
|---|---|---|
| `cache_candidates.py` -> `{name}.npz` | coords, src, tgt, **prob**, dist | the old replay line |
| `cache_train.py` -> `detcache_*.npz` | `c{t}`, `s{t}` only | **ranker_fine, div_train, everything current** |

So the current best pipeline feeds the ILP a gradient-boosted tree over 18
hand-made geometric numbers and **discards the end-to-end-trained edge
head entirely** (`edge_weight = -EdgeAttr("edge_prob")` where edge_prob is
the ranker's output). That replacement was a net win -- 0.876 -> 0.921 --
but "geometry alone beats the edge head alone" is not "the edge head has
nothing to add".

**NOT free.** The cached probs in `{name}.npz` are indexed to the OFFICIAL
detector's nodes; the pipeline now runs xl detections, which are different
nodes. Using `edge_prob` as a ranker feature needs the official edge head
re-run over the xl node set -- a GPU inference pass, and the head needs
image features at node locations, so it is not a pure cache join. Size the
pass before committing to it.

## 2026-08-24 -- ⭐⭐ div-gate: the offline AP gain does NOT survive contact with the pipeline

`div_gate.py` -> `biohub-div-gate`, 53 min on a T4. Solve each holdout
dataset exactly as the recorded pipeline does (div_w 0.55, global), score
every resulting fork with the FULL division model from
biohub-div-kinematics, sweep a prune threshold, rescore.

**Anchor reproduces.** prune 0% gives TP/FP/FN 5/23/7, div_J 0.1429,
adj_J 0.9220, score 0.9363, against the recorded 5/26/7 / 0.1316 / 0.9214
/ 0.9345 at div_w 0.50 -- identical TP and FN, three fewer FP, which is
the direction a higher div_w should move. The wiring is correct.

| prune | forks cut | TP | FP | FN | div_J | adj_J | score |
|---|---|---|---|---|---|---|---|
| **0%** | 0 | **5** | 23 | 7 | **0.1429** | 0.9220 | **0.9363** |
| 25% | 138 | 4 | 22 | 8 | 0.1176 | 0.9222 | 0.9340 |
| 50% | 276 | 2 | 10 | 10 | 0.0909 | 0.9226 | 0.9317 |
| 75% | 414 | 1 | 4 | 11 | 0.0625 | 0.9230 | 0.9292 |
| 90% | 497 | 1 | 3 | 11 | 0.0667 | 0.9229 | 0.9296 |

**Monotone downhill. Doing nothing wins.** Pruning removes true divisions
at least as fast as false ones: the bottom 25% of forks by `p_div` holds
20% of the TPs but only 4% of the FPs -- *anti*-informative. By the 50%
cut it is 60% of TPs against 57% of FPs, i.e. indistinguishable from
cutting at random.

### Why the offline number did not transfer -- the lesson

div-kinematics measured AP 0.0812 and 1.8 FP per TP **over every candidate
triple `build()` emits** -- every neighbour pair of every parent, which is
overwhelmingly obvious non-divisions. This kernel scores **only the 553
forks the pipeline already chose**, which have passed the ILP's division
weight or the orphan patch's strict geometric filters. Those are the hard
cases; the easy negatives were gone before the model ever saw them.

**The offline gain was the model re-separating negatives the pipeline had
already excluded.** Measure a classifier on the population the
intervention will actually act on, not on the population it was trained
on. This is now the second time an offline win evaporated in the pipeline
(the first: scale-invariant edge features).

### What survives and what does not

- STILL TRUE: forward features carry real signal on the full candidate
  set. `sep_slope` 1.155 vs 0.082 is a measurement, not a model artefact.
- DEAD: `p_div` as a post-hoc filter on selected forks.
- DISCOURAGED, not disproven: re-ranking the orphan patch's *proposals*
  by `p_div` instead of by `d_pq + 0.15*d_s`. That acts on a larger pool
  than the accepted forks, so it is not strictly the same population --
  but the model showing zero separation on the accepted subset is weak
  evidence it would separate the pool either.
- Ceiling unchanged: FN 7 is 2 lost to detection, 3 outside `GATE_UM`, 2
  below `ILP_FLOOR`. Even perfect FP removal caps div_J at 5/12 = 0.417,
  worth +0.027. **That ceiling needs a model that can separate; this one
  cannot.**

⚠ **Five true divisions.** Every row rests on TP counts of 1-5, far under
anything resolvable alone. The monotone trend across five points is the
evidence here, not any single delta.

**This is more weight behind the conclusion already on record below: the
divisions are lost upstream, and the division line is close to exhausted.**

## 2026-08-24 -- ⭐⭐ division kinematics: the DAUGHTERS' future works, the mother's past does not

`div_kinematics.py` -> `biohub-div-kinematics` v2, ~23 min on a T4. One
feature build, four column views. Extends the diagnosis already on record
in the gate-round-3 entry below: "the model has no way to recognise the
rare configuration where a long edge IS a second daughter, because it
scores each edge independently and a division is a property of the
triple". The triple model was the first half; **time is the second.**

Trained on Zebrahub only (2001 pos), validated on held-out 6bba (83 pos),
same protocol as div-train v3.

| arm | cols | AUC | AP | FP per TP @ 10% recall |
|---|---|---|---|---|
| BASE (div_train's 25) | 25 | 0.8275 | 0.0486 | **6.9** |
| + mother's PAST | 35 | 0.8289 | 0.0375 | 8.8 |
| + daughters' FUTURE | 35 | 0.8365 | **0.0812** | **2.5** |
| FULL | 49 | 0.8198 | 0.0747 | **1.8** |

**The user's call, and it was right against mine.** The session had
front-loaded the backward half because it is cheap (before t the mother
is ONE cell, no chicken-and-egg). The user pushed back -- "what if the
daughters' behaviour is the more suspicious one" -- and the data agrees:
backward does NOTHING (AP gets *worse*, 0.0486 -> 0.0375), forward nearly
doubles AP and cuts the false-division price 2.8x.

**Read AP and FP/TP, never AUC, on this problem.** Positives are 0.25%.
AUC moves 0.83 -> 0.84 across every arm while AP doubles, and FULL has
the WORST AUC of the four yet the BEST FP/TP. An AUC-only report would
have picked the wrong arm.

### The physics, checked with no model in the way (medians, 6bba)

| column | true div | other | ratio |
|---|---|---|---|
| **fwd `sep_slope`** (um/frame) | 1.155 | 0.082 | **14.1** |
| fwd `sep_end/0` | 1.431 | 1.031 | 1.39 |
| fwd `score_min` | 0.396 | 0.289 | 1.37 |
| **coup `cent_resid`** (um) | 2.872 | 5.053 | **0.57** |
| back `decel` | 0.905 | 0.934 | 0.97 |
| back `v_last` | 2.254 | 1.861 | 1.21 |

- **Real daughters separate at 1.16 um/frame; random pairs at 0.08.** A
  14x gap, and the single cleanest division signal this project has
  found.
- **The centroid-momentum claim holds**: at a true division the pair's
  centroid lands 2.87 um off the mother's extrapolated position, against
  5.05 um for a spurious pair. `cent_resid` is also the top permutation
  column (+0.0058, 4x the next).
- **The mitotic slow-down does not exist in this data.** `decel` is 0.905
  vs 0.934 -- indistinguishable, and `v_last` runs the WRONG way (dividing
  mothers are slightly *faster*). Most likely mitotic rounding is short
  against the frame interval and simply is not sampled. That hypothesis
  is DISPROVEN, not untested.
- Two predictions that failed: `sep_min/0` ("two cells merely passing each
  other converge first") ratio 1.05 and NEGATIVE permutation importance;
  `surv_both`/`surv_gap` ("one member vanishes") are 5.0/0.0 in both
  classes at FWD_K=5 -- nothing vanishes, so survival discriminates
  nothing.

Low permutation importance here means REDUNDANT, not useless: `sep_slope`
scores +0.0002 despite the 14x raw separation, because `sep_end/0` carries
the same information inside the FULL model.

### Where this cashes out

The metric's marginal budget is ~7 FP per TP. **BASE sits at 6.9 -- right
on the edge, which is why divisions have never been worth taking.** FWD
buys them at 2.5 and FULL at 1.8, roughly 4x cheaper. Against the
forensics funnel below (12 GT divisions: 2 lost to detection, 3 outside
the gate, 2 below the floor, 5 taken), the 2 below-floor ones are the
direct target, and the 3 gate ones become reachable if the gate widens
AND the classifier can now hold them above the floor -- which is exactly
the combination round 3 lacked.

**Not yet done: this is offline AUC/AP, not div_J.** Wiring the FULL model
into the pipeline and scoring div_J on the holdout is the real test.

### ⚠ `hash(name)` makes the negative sampling irreproducible across runs

`build()` seeds its negative thinning with
`np.random.default_rng(hash(name) % 2**31)`. Python randomises `hash()`
per process unless `PYTHONHASHSEED` is set, so every run keeps a
different negative sample: this run got 49586 zebrahub / 46770 competition
triples where div-train v3 got 49735 / 46631, and BASE scored 0.8275 here
against v3's recorded 0.8083. **Arm-to-arm comparisons inside one run are
valid (all arms see identical rows); any cross-run AUC comparison in this
project carries ~0.02 of pure sampling noise.** Fix with a stable seed
(`zlib.crc32(name.encode())`).

## 2026-08-24 -- scale-invariant edge features: NO EFFECT, and the -0.73 is not in the ranker

`ranker_scale.py` -> `biohub-ranker-scale`, 8.6 min on a T4. One feature
build, three column views, so the arms can differ ONLY in which columns
the model is allowed to see. The 18 base columns are a transplant of
yusuketogashi's public 0.915 ranker (`public-solutions-recon.md`,
addendum 2026-08-22b) -- nobody here chose them.

| arm | cols | what |
|---|---|---|
| BASE | 18 | the transplant, unchanged |
| EXT | 28 | + local spacing, `d/sp`, `md/sp`, `gain/sp`, `xy/sp`, `z/sp`, `vel/sp`, and density at an adaptive 2x-spacing radius instead of the hardcoded 7 um |
| PURE | 17 | scale-free only: no absolute um, no fixed radius, not even the spacing itself |

Local scale = median distance to the 5 nearest neighbours. Verified
before the push: a 6 um lattice gives 6.0, the same lattice at half size
gives 3.0, coincident detections do not collapse it to zero.

**Every measurement says the feature set does not matter.**

| | BASE | EXT | PURE |
|---|---|---|---|
| AUC, `ranker_fine`'s own split | **0.9884** | 0.9886 | 0.9875 |
| corr(per-dataset AUC, density) | +0.053 | +0.021 | -0.083 |
| sparse-60% -> dense-40% AUC drop (bigger = worse) | 0.0213 | 0.0229 | 0.0258 |
| holdout linker-only adj_J | **0.9293** | 0.9286 | 0.9269 |
| corr(adj_J, density) | **-0.733** | -0.736 | -0.739 |

BASE reproduced the recorded 0.9884 **exactly**, so the reproduction is
faithful and the comparison is trustworthy. The adj_J spread across arms
is 0.0024, below the ~0.003 resolvable floor. Scale-invariant features
are if anything very slightly WORSE, and PURE is worst on every single
line -- dropping absolute micrometres costs a little and buys nothing.

### The useful part: where the -0.73 is NOT

`corr(adj_J, density) = -0.733` reproduces the recorded -0.68 closely, so
the penalty is real and it survives every reparametrisation tried here.
But `corr(AUC, density)` is **+0.05, i.e. flat**. The ranker scores edges
just as well when crowded.

The clearest single case is `6bba_3abfe10a`, 692 cells/frame, the densest
holdout dataset by 3x:

- AUC 0.9773 -- mid-pack, indistinguishable from the sparse datasets
- adj_J 0.7788 -- worst of the twelve by 0.15

**So the density penalty is downstream of edge scoring.** AUC ranks all
pairs; adj_J depends on the top-1 choice per cell. As candidates per cell
grow, a good ranking still leaves many near-ties for 1:1 Hungarian at a
fixed `p_min` to break. Detection is the other suspect and the more
likely one -- `corr(edge J, N/N_true) = -0.64` is already on record.

**Do not re-try this by adding more geometry to the ranker.** Two things
that follow instead:

- the assignment step rather than the score: candidate count per cell, or
  a density-dependent `p_min`
- detection under crowding, which adj_J is bounded by

### Why the reparametrisation was nearly a no-op

Density spans 15x across this train set (46 -> 692 cells/frame) but local
spacing spans only 2.1x (10.03 -> 21.23 um) -- as it must, since spacing
goes as density^(-1/3) and 15^(1/3) = 2.47. Dividing by a quantity that
moves 2.1x cannot undo a 15x change in the thing that actually matters.
The premise "every distance feature is absolute micrometres" was true;
the inference that this is what costs us was not.

Artefacts: `edge_ranker_{base,ext,pure}.pkl` in the kernel output. BASE is
equivalent to the shipped `edge_ranker.pkl`; the other two are not worth
keeping.

## 2026-08-24 -- windowing reverted; the seam hypothesis is disproven

Three LB points on the same pipeline, one variable:

| ref | solve | LB | total_bytes (hidden test) |
|---|---|---|---|
| 55707206 | global | **0.921** | 262.6 MB |
| 55709347 | 25-frame windows | 0.920 | 289.6 MB |
| 55716291 | windows + seam constraint | 0.920 | 289.3 MB |

The windowed runs emit **10% more output on the hidden test** while
emitting 2% LESS on the visible 4 -- the visible 4 do not reproduce the
behaviour, so they could not have caught this.

The seam fix propagated the previous window's REFUSALS across the shared
frame (the half of ultrack's scheme the first version omitted, since
sharing a frame stops edge sets colliding but does not stop window k+1
treating a node window k rejected as a fresh track start). It fired --
25 to 170 candidate edges cut per movie -- and changed **nothing**:
bytes 289.6 -> 289.3, score unchanged. **The hypothesis is disproven.**

Untested alternative worth trying if anyone revisits this: the ILP
charges `disappearance_weight` 1.5 at every track end, and under
windowing every track appears to END at each seam, so the penalty is
paid W times instead of once. That biases the solver toward linking
more aggressively, which fits +10% output.

**The decision, though, is simpler than the diagnosis.** Windowing
existed to buy margin against the 12 h cap. The global solve
COMPLETED on the hidden test at 0.921. The risk never materialised, so
the 0.001 windowing costs buys nothing. Reverted.

## 2026-08-24 -- div-train v1/v2 INVALID: a time compared against a row index

Two runs produced numbers that must not be used. The Zebrahub division
extraction was broken by:

```python
if kk not in first or tt < first[kk]:     # tt is a TIME
    first[kk] = r                          # first[kk] is a ROW INDEX
```

so "each track's earliest node" resolved to arbitrary rows and
parent-child pairing collapsed. Result: **26 of 41 crops reported no
division at all, and Zebrahub yielded 68 positives** where hundreds of
thousands were expected. Every AUC printed by those runs (0.2255,
0.2161, 0.4332, 0.3417, 0.3145) is meaningless -- and AUC below 0.5 was
itself the tell.

Two further design faults found at the same time:

- **4,632,084 competition triples for 101 positives.** A 0.002% positive
  rate; the model sees nothing. v3 keeps every positive and thins
  negatives to 1%.
- **The split was unusable.** Training on 44b6 alone gives **18
  positives**. v3 asks the question that actually matters -- can
  Zebrahub carry training by itself? -- and validates on 6bba (83
  positives), which enters no training set.

The one solid number from the run: **124 of the 151 GT divisions land on
detections** once all 199 datasets are cached, confirming the earlier
projection of ~127.

## 2026-08-24 -- the division data question, measured end to end

The linker is finished (forensics entry below): the solver already takes
every division the candidate set contains. Two rounds then tried to widen
the candidate set and both failed -- but not for the reason they looked
like they failed for.

### `GATE_UM`: two rounds, and the first one was confounded

Round 1 (`biohub-gate-sweep`) held div_w at 0.55 while widening the gate,
and divTP COLLAPSED 5 -> 2. That looked like the gate failing. It was
not: a wider gate floods training with distant negatives (positives
46.8% -> 21.2% -> 11.4%), which drags the whole probability scale down --
median probability fell **90x**, 0.1196 -> 0.0013. div_w and ILP_FLOOR
are absolute thresholds ON that scale, so holding them fixed made them
far stricter. **The gate was never tested.** Recorded as untested, not
disproven, precisely so nobody closes the line on 0.9301.

Round 2 (`biohub-gate2`) trained once per gate and swept div_w on the
cached probabilities. Now it is a fair test, and the gate still loses:
best at gate 14 is 0.9357 (floor 0.03, div_w 0.25) against 0.9382 at
gate 10. The funnel says why: gate 14 closes stage 2 (gate 3 -> 0) but
those divisions land in stage 3 (floor 2 -> 4) and REACHABLE only moves
5 -> 6. At floor 0.03 divTP does reach **6** -- one more than gate 10
ever managed -- but it costs **+31 division FPs for that one TP**,
against a budget of ~7 (the marginal rate the metric allows).

Round 3 (`biohub-zgate`) loosened Z ONLY, an ellipsoid instead of a
sphere, because the missed divisions were z-dominated: volume grows as
GZ/GYX = 1.4 instead of (14/10)^3 = 2.74. The isotropic control
reproduced 0.9382 exactly, so the implementation is sound. It still
fails, and the funnel is decisive: gate 3 -> 2 -> 1 as GZ rises, floor
2 -> 3 -> 4, **REACHABLE never leaves 5**. Widening the gate moves a
division from "outside gate" to "below floor". The ranker scores those
long-z edges low -- and it is RIGHT to: hengck23 verified independently
on the host baseline that "long links are almost wrong". The binding
constraint is not the gate, it is that the model has no way to recognise
the rare configuration where a long edge IS a second daughter, because
it scores each edge independently and a division is a property of the
TRIPLE (parent, d1, d2).

### Why it cannot learn that: 151 divisions, from 2 embryos

`biohub-div-census`: **151 GT divisions in the whole 199-dataset train
split**, 87 datasets have any, median 1, max 5. Only 31 sit in the 44
cached datasets; 26 of those are fully detected. Caching the rest yields
~127. A forum comment pins the structure: the training set is **2
embryos**, split **26 (44b6) + 125 (6bba)** -- which matches our census
exactly. So the data is not merely scarce, it has no independent
replication.

For scale: the same forum thread offers a free CC0 synthetic set with
**165,267 divisions**, and Zebrahub has far more.

### Zebrahub divisions are real: 468,617 of them

`biohub-zh-divisions` validated every candidate (daughters born on the
same frame, both surviving >= 3 frames, non-zero separation):

| movie | raw | validated | sisterSep | z share | angle |
|---|---|---|---|---|---|
| ZSNS001 | 635,041 | 358,314 (56%) | 7.37 | 0.84 | 138 |
| ZSNS001_tail | 36,878 | 36,246 (98%) | 6.87 | 0.78 | 149 |
| ZSNS003 | 15,868 | 12,719 (80%) | 6.03 | 0.67 | 148 |
| ZSNS004 | 30,278 | 24,866 (82%) | 7.78 | 0.65 | 150 |
| ZSNS005 | 46,026 | 36,472 (79%) | 7.60 | 0.82 | 154 |

**Fragmentation is ruled out**: across 764k candidates, every single
parent has exactly 2 children -- a re-numbered broken track would give
3 and 4 as well. ZSNS001's 2.9% rate was the original suspicion; it is
simply a longer, denser movie.

### Transfer, measured feature by feature (not argued)

`biohub-zshare` computed competition geometry from GT alone -- no
detector, so no survivorship -- over all 151 divisions, split by embryo:

| | n | sisterSep | z share | angle |
|---|---|---|---|---|
| 44b6 | 26 | 8.98 | 0.43 | 135 |
| 6bba | 125 | 11.47 | 0.55 | 140 |
| ALL | 151 | **10.57** | **0.53** | 138 |

**CORRECTION.** An earlier entry called sister separation a match
(Zebrahub 7.37 vs "competition 7.24"). That 7.24 was the synthetic
dataset author's calibration, not our measurement. Ours is **10.57 um**.
Zebrahub is ~30% SMALLER, not matching.

`biohub-density` then explained that gap, testing the user's hypothesis
that Zebrahub is simply denser. Normalising each separation by the local
cell spacing (median distance to the 5 nearest detections/nodes):

**sisterSep / local spacing: competition 0.68-0.85, Zebrahub
0.62-0.79 -- the distributions overlap heavily.**

`SUPERSEDED 2026-08-24`: an earlier version of this entry claimed the
two agree "to 2%" (0.72 vs 0.71). That number came from pairing the
competition's COUNT-weighted value against Zebrahub's
EQUAL-PER-MOVIE value -- the Zebrahub side was capped at 4000 divisions
per movie, so ZSNS001's 358k of the 469k total counted the same as
ZSNS003's 13k. Weighted consistently the two sides cross over:

| | count-weighted | equal per unit |
|---|---|---|
| competition (0.85 / 0.68) | 0.71 | 0.77 |
| Zebrahub (0.79/0.72/0.62/0.72/0.69) | 0.77 | 0.71 |

Either way the gap is 7-8%, not 2%. **The defensible claim is overlap,
not equality** -- and 7-8% still sits inside the competition's own
between-embryo spread (0.68 to 0.85 is 20%), so the conclusion that
Zebrahub is another embryo domain rather than another problem survives;
only its precision was overstated.

Note the per-movie cap is not itself a flaw: **how much of each movie
enters training is a free parameter**, so equal sampling is a legitimate
design. The error was using a training-design weighting to make a
DESCRIPTIVE claim about the domains. Consequence for the build: do not
try to derive the right mixture analytically -- **sweep it** (per-movie
sampling, and the Zebrahub:competition weight ratio, currently 3700:1)
and let AUC on the 127 in-domain divisions choose.

"Daughters are born roughly 0.7 nearest-neighbour distances apart" holds
as an order-of-magnitude invariant on both sides. The 30% raw gap was
scale, not geometry. Caveat: the WITHIN-dataset correlation of spacing
against separation is ~0 (+0.11, +0.04), so this is a between-dataset
scaling, not a per-division crowding force.

**A second CORRECTION**: an isotropic direction gives median
|dz|/||d|| = **0.5** exactly (|cos theta| is uniform on [0,1]); an
earlier entry said 0.58, which is the RMS. So competition at 0.53 is
essentially ISOTROPIC and Zebrahub at 0.65-0.84 is genuinely z-biased.

Per-axis normalisation was tried next and **did not** reconcile them,
against expectation:

| | packing sz/sxy | \|dz\|/sz | \|dxy\|/sxy |
|---|---|---|---|
| 44b6 | 0.55 | 0.32 | 0.61 |
| 6bba | 0.70 | 0.54 | 0.56 |
| competition | 0.64 | 0.51 | 0.56 |
| Zebrahub | 0.52 | **0.75** | **0.33** |

Packing anisotropy is comparable (0.64 is exactly the isotropic
expectation, since E|dz| = r/2 and E||dxy|| = pi*r/4). But the division
AXIS differs and survives normalisation. Note the competition's own two
embryos disagree -- 44b6 is xy-dominated, 6bba balanced, Zebrahub
z-dominated: **a feature that is unstable between the two training
embryos has no claim to stability on unseen ones**, and train/test are
embryo-disjoint by the rules.

### The variance decomposition that answers "is Zebrahub another world"

| quantity | between competition embryos | between Zebrahub embryos | across datasets |
|---|---|---|---|
| z share | 0.12 | 0.19 | 0.22 |
| sister separation | 2.5 um | 1.75 um | ~3 um (scale; normalises away) |
| daughter angle | 5 deg | 16 deg | ~5 deg |

Every cross-dataset gap is the size of an embryo-to-embryo gap. Zebrahub
is one more embryo domain, not a different problem -- and we are required
to generalise across embryos regardless.

**On the local-to-LB gap**, the user's reading was that we score better
on sparse data (true: `corr(edge J, density) = -0.68`, 0.968 at 50
cells/frame vs 0.771 at 519) so the hidden test must be denser. Partly:
the two families show -0.042 and -0.015 against the SAME hidden test, so
most of the larger gap is holdout memorisation by the official weights,
not density. The xl family's -0.015 is the part density can explain, and
it just predicted 0.921 correctly.

## 2026-08-23 -- SUBMITTED ref 55709347: windowed ILP. Better on both rulers, and it fits the clock.

`biohub-submit-w25` = ref 55707206 with a single change, the ILP solved
in 25-frame windows overlapping by one frame.

**Runtime, measured on the visible 4 (this was the open question):**

| dataset | pairs | feats | solve | total | was |
|---|---|---|---|---|---|
| 44b6_0113de3b | 55 581 | 3 s | 19 s | 111 s | 120 s |
| 44b6_0b24845f | 73 589 | 5 s | 26 s | 126 s | 145 s |
| 6bba_05b6850b | 8 711 | 1 s | 4 s | 88 s | 90 s |
| 6bba_05db0fb1 | 272 577 | 16 s | **100 s** | **222 s** | **485 s** |

The densest movie's solve fell **390 s -> 100 s (3.9x)**: windowing pays
most exactly where the ~n^1.5 blow-up hurts. Average 137 s/dataset
against 210 s, so **~7.7 h for ~200 volumes**, a 36% margin under the
12 h cap where the global solve projected 11.8 h. **The runtime line is
closed.**

**Three-way paired score on the visible 4** (`biohub-score-subs`, all
three finished CSVs against train GT -- movies that are not the
holdout, two of them from the `44b6` family the holdout lacks):

| | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl 0.917 pipeline | 0.8678 | 0.8678 | 0.0000 | 0/5/3 |
| ranker, global ILP | 0.8855 | 0.8855 | 0.0000 | 0/11/3 |
| **ranker, window 25** | **0.8932** | **0.8932** | 0.0000 | 0/12/3 |

w25 - xl0917 = **+0.0254**; w25 - global = **+0.0077**. Windowing gains
MORE here than on the holdout (+0.0014), so it is not a holdout artefact.

Still carried: div_J is 0.0000 for all three -- those 4 movies hold 3 GT
divisions and nobody catches any. **Every visible-4 delta is pure
adj_J**; the division half of the holdout result remains single-source.

## 2026-08-23 -- rule 1 audit: four silent degradations found and given a voice

The user reaffirmed engineering rule 1. Audited this session's scripts;
four paths dropped data without saying so, all now counted and warned:

| where | what was silent |
|---|---|
| `solve_edges` | a window producing no edges left those frames UNLINKED |
| geff edge filter | `if a in rm and b in rm` dropped edges whose endpoint lost its node |
| `_emit` | self-loops, missing nodes and duplicates all `continue`d together |
| `score_subs` | CSV edges naming an absent node id -- **in the SCORER**, so a config could be judged on a silently low score |

The last is the dangerous one: a silent drop there biases a measurement
rather than an output, and could get a better configuration rejected.

`_emit` was given a COUNT ONLY -- its three drops are xl-repro's verbatim
behaviour that produced 0.917, so it now speaks without changing what it
does. Note the running `gate_sweep` and `score_subs` jobs were launched
BEFORE this patch; if their numbers look odd, suspect this first.

## 2026-08-23 -- ⭐⭐⭐ FORENSICS: the linker is finished. Every remaining division is lost upstream.

`biohub-div-forensics` walked all 12 holdout GT divisions down the funnel.

| stage | what stops it | count |
|---|---|---|
| 0 | parent undetected | 1 |
| 1 | a daughter undetected | 1 |
| 2 | **pair outside `GATE_UM` (10 um)** | **3** |
| 3 | probability below `ILP_FLOOR` (0.10) | 2 |
| 4 | reachable -- the solver's call | 5 |

**divTP at div_w 0.55 is 5, and stage 4 is 5. The solver takes ALL of
them.** FN 7 equals the seven unreachable ones exactly. That closes the
question the twelve-point sweep raised: division FN never fell below 6
because the linker had already extracted 100% of what the candidate set
contained. **No linker-side knob -- div_w, windowing, calibration --
can move div_J further.** Stop tuning it.

Per-division detail (parent match distance, and for stage 1 the nearest
detection of any score to the missed daughter):

```
6bba_7af54fde t=55 stage=0  pd=5.16
6bba_afb141ff t=48 stage=1  pd=1.15  daughter miss 5.47 um, nearest det
                                     5.47 um scoring 0.412
6bba_268e1230 t=24 stage=2  pd=2.12
6bba_2312ac41 t=10 stage=2  pd=3.52
6bba_7af54fde t=52 stage=2  pd=1.55
6bba_3abfe10a t=81 stage=3  pd=3.16  min pair prob 0.016
6bba_5c824876 t=2  stage=3  pd=0.33  min pair prob 0.080
6bba_2312ac41 t=19 stage=4  prob 0.393    6bba_afb141ff t=47  0.711
6bba_afb141ff t=32 stage=4  prob 0.215    6bba_afb141ff t=28  0.605
6bba_aeee7805 t=13 stage=4  prob 0.797
```

Note stage 1 is not really a detection failure: a detection scoring
0.412 (well over `UNET_THRESH` 0.15) sits 5.47 um away, just outside
`MATCH_UM` 5.0. It is a matching-radius near-miss.

### The lever: `GATE_UM`, worth roughly twice the whole ranker line

Stage 2 is the biggest actionable bucket -- 3 divisions where parent AND
both daughters are detected and the pair simply never enters the
candidate set. Sister separations read off the strips:

```
268e1230 t=24  11.55 um     2312ac41 t=10  16.05 um
3abfe10a t=81  11.98 um     5c824876 t=2   11.55 um
7af54fde t=52  12.90 um     7af54fde t=55  12.19 um
afb141ff t=48   7.85 um
```

Against a 10 um gate, and the strips repeatedly print `OUT OF SLAB` with
z jumping 8-10 planes: **divisions here separate along Z**, the coarse
axis at 1.625 um/plane (4x the xy pitch), so daughters gain physical
distance fast. The gate was sized for ordinary motion (the 6bba median
displacement p99 is 7.81 um) and divisions are not ordinary motion.

Arithmetic if raising the gate recovers all three: TP 5 -> 8, FN 7 -> 4,
div_J 5/35 = 0.1429 -> 8/35 = 0.2286, worth **+0.0086** on the composite
-- nearly twice everything the ranker line has bought (+0.0045).

**Do not just raise the constant.** `pair_features` includes `rank`
within the source's candidate list and both candidate counts; widening
the gate shifts those feature distributions, so **the ranker must be
retrained at each gate value** or its probabilities are read off a
distribution it never saw. Retraining costs 170 s.

Design for the next kernel: sweep `GATE_UM` 10/14/18 with a retrain at
each, and report the funnel, the composite, and the solve time (a wider
gate means more candidate pairs, so it buys divisions and pays in
runtime -- exactly the trade the 12 h cap cares about).

### Timing split: the solve is 88% of linking, features are 12%

| dataset | pairs | vel | feats | predict | solve |
|---|---|---|---|---|---|
| 6bba_268e1230 | 6 938 | 0.1 s | 0.4 s | 0.3 s | 4.3 s |
| 6bba_2312ac41 | 18 484 | 0.2 s | 1.0 s | 0.3 s | 11.2 s |
| 6bba_283bf9f1 | 39 027 | 0.5 s | 2.0 s | 0.4 s | 22.6 s |
| 6bba_3a1849c2 | 10 968 | 0.1 s | 0.6 s | 0.2 s | 5.9 s |

Feature side 6 s against 44 s of solve. **Windowing attacks the right
88%.** Projection for the hidden test: the submission run's 210 s/dataset
splits as detect 86 s + features ~15 s + solve ~109 s; at the measured
2.08x the solve drops to ~52 s, giving ~153 s/dataset and **~8.5 h for
~200 volumes** against the 12 h cap. Comfortable-ish, not generous, and
`biohub-submit-w25` will replace this estimate with a measurement.

## 2026-08-23 -- ⭐⭐ windowing the ILP is free speed AND a better score; calibration is dead

`biohub-ranker-wincal`, two independent lines in one kernel.

### W line -- window the ILP in time (uncalibrated, div_w 0.55)

| window | score | adj_J | div_J | ILP forks | link | speedup |
|---|---|---|---|---|---|---|
| global | 0.9368 | 0.9233 | 0.1351 | 111 | 326 s | 1.00x |
| 50 | 0.9368 | 0.9230 | 0.1389 | 111 | 213 s | 1.53x |
| **25** | **0.9382** | 0.9235 | **0.1471** | 108 | **157 s** | **2.08x** |
| 10 | 0.9343 | 0.9214 | 0.1290 | 88 | 126 s | 2.59x |

**Window 25 is both faster and better** (+0.0014 over the global solve at
2.08x). W=10 cuts too deep and loses real context. Likely mechanism for
the gain: with `disappearance_weight` 1.5 the global solve is pushed to
build very long tracks; a window bounds how far that pressure travels --
and it is div_J that improves (0.1351 -> 0.1471, division FP 25 -> 22),
i.e. fewer wrong long-range commitments.

Scheme (copy it, do not reinvent): window k covers frames [a, b] and
k+1 covers [b, c] -- they SHARE ONE FRAME but never share an EDGE, since
k owns a->a+1 .. (b-1)->b and k+1 owns b->b+1 .. So merged edge sets
cannot conflict and no reconciliation pass is needed. Same as ultrack's
`window_size` / `overlap_size` (their default overlap is 1).

**Trap paid for in advance:** mapping solved nodes back to global indices
must be done BY COORDINATE, never by `node_id` -- node_id is not a stable
handle across the solve (progress.md 02f7c8b).

Caveat on the control: `wglobal` reads 0.9368 where the fine sweep's
div_w 0.55 read 0.9363. Not noise -- `run_config` assembles nodes from
ALL seeds and lets `_short_filter`/`_emit` prune, where the B line built
them from the ILP's kept nodes. The window comparison is paired inside
this kernel, so it stands; do not mix these two numbers across kernels.

### C line -- probability calibration: HYPOTHESIS WRONG, do not retry

The theory was that the ranker's labels (BOTH GT children positive)
inflate second-daughter probabilities, and that div_w 0.55 is paying for
that bias. Isotonic regression fitted on the held-out 10%:

```
raw mean 0.4191 -> cal 0.4168;  frac>0.55  0.4214 -> 0.4118
```

**The probabilities were already calibrated.** Isotonic moved the mean by
0.0023. And the calibrated sweep is monotone increasing in div_w and
never beats uncalibrated -- 0.9258 / 0.9286 / 0.9311 / 0.9343 at div_w
0.05 / 0.15 / 0.30 / 0.55, against 0.9368 uncalibrated at 0.55. Isotonic
flattens probabilities into steps, creating ties the ILP then breaks
arbitrarily; that is the likely cost. **Line closed.** The ultrack
contrast (their `division_weight` default is -0.001) therefore needs a
different explanation -- most likely their candidates are
segmentation-hypothesis based and mutually exclusive by construction.

### ⭐ The structural finding, worth more than either line

Across ALL twelve sweep points -- 8 uncalibrated, 4 calibrated, forks
ranging 0 to 1847 -- **division FN never drops below 6**. divTP moves
only between 4 and 6 while division FP explodes.

The divisions we miss are **not in the candidate set at all**; the solver
is not refusing them. That kills the "7 FPs per TP is a bargain" line of
attack (user's marginal analysis, which is arithmetically right): the
exchange rate is fine, there is simply nothing to buy. Anything that
moves div_J has to act UPSTREAM -- the candidate gate `GATE_UM`, the
detection threshold (a just-divided second daughter is small and dim and
plausibly sits under `UNET_THRESH` 0.15), or a purpose-built division
detector. `biohub-div-forensics` walks every GT division down the funnel
to find which stage loses them.

### Metric note that makes over-forking cheap (user's observation, verified)

`reference.md#the-metric`: a fork lying entirely among unannotated cells
enters no FP set. Measured from the div_w 0.55 -> 0.20 step: 913 extra
forks cost only -0.0027 adj_J, which at ~10.7k holdout edges is roughly
**30 visible edge FPs -- about 97% of the extra forks are invisible to
the metric**. But the div_J denominator is the real bill: the same step
cost -0.0056 there, twice the edge damage. So forks are nearly free on
edges and not free on divisions. (adj_J is a weighted per-dataset average
with a node-count penalty, so treat the 30 as an order of magnitude.)

## 2026-08-23 -- SUBMITTED ref 55707206: ranker + capacity-2 ILP linker

Kernel `biohub-submit-ranker` v2. Single variable against LB 0.917
(ref 55690436): the linker. Detection side and post-link patches are
unchanged; the Hungarian is replaced by ranker probabilities feeding a
capacity-2 tracksdata ILP at division weight 0.55.

**Independent confirmation before submitting.** `biohub-score-subs`
scored BOTH finished CSVs against train GT on the same visible 4 -- a
paired comparison on movies that are not the holdout, two of them from
the `44b6` family the holdout does not contain:

| | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl 0.917 pipeline | 0.8678 | 0.8678 | 0.0000 | 0/5/3 |
| **ranker + cap-2 ILP** | **0.8855** | **0.8855** | 0.0000 | 0/11/3 |
| delta | **+0.0178** | +0.0178 | 0 | |

Four times the holdout delta, on different movies -- different data, so
the magnitudes are not comparable, but the sign is confirmed away from
the holdout, which is the thing the family-calibration rule says to
worry about.

**Limit on that evidence, carried deliberately:** div_J is 0.0000 for
BOTH pipelines here. Those 4 movies hold only 3 GT divisions and neither
pipeline caught any (ranker 11 division FPs vs xl's 5, but with 0 TP the
division term is 0 either way). So **the +0.0178 is entirely adj_J**;
the division half of the holdout result (div_J 0.1429 vs 0.1333) has NOT
been reproduced outside the holdout and remains single-source evidence.

The scoring kernel then died on a cosmetic per-dataset print
(`per_sample_metrics` returns no "score" key -- that appears only after
`summarise`); every number above had already been emitted. Not worth a
re-run, but fix `score_subs.py` before reusing it.

## 2026-08-23 -- the submission kernel runs, and the ILP does not fit the 12 h cap

`biohub-submit-ranker` v2 COMPLETE: offline env works (pack wheels,
tracksdata, `predict_unet_transformer` from the pack, ranker pkl via
kernel source), all three xl models load, `submission.csv` written with
the right schema. v1 had died on `edge_ranker.pkl` holding
`{"model": clf, "version": ...}` rather than a bare model -- the load
now unwraps and asserts `predict_proba` at load time, so that class of
error surfaces in 40 s instead of after a 10 min model load.

Counts against the 0.917 pipeline on the same visible 4 (a real paired
check that the graft is not producing well-formed garbage):

| dataset | xl0917 nodes/edges | ranker nodes/edges | ILP forks | runtime xl -> ranker |
|---|---|---|---|---|
| 44b6_0113de3b | 23878 / 23152 | 23794 / 23048 | 5 | 82 s -> 120 s |
| 44b6_0b24845f | 28147 / 26186 | 27599 / 25862 | 20 | 87 s -> 145 s |
| 6bba_05b6850b | 6135 / 5906 | 6090 / 5894 | 7 | 74 s -> 90 s |
| 6bba_05db0fb1 | 68259 / 65947 | 66813 / 64421 | 8 | 102 s -> **485 s** |

Nodes -1.7%, edges -1.6% -- the small drop the ILP's node-dropping
predicts. Forks per dataset match the holdout's 111/12.

### ⚠ THE RUNTIME PROBLEM (read before touching the submission kernel)

The hidden test is **~199-200 volumes** (`competition-brief.md:189`) and
the cap is 12 h. Measured here: **210 s/dataset average** -> 200 x 210 s
= **11.7 h + 10.6 min setup**, i.e. flush against the cap. The 0.917
pipeline runs 86 s/dataset (~5 h), so the whole excess is the linker.

Worse, the linking cost is **superlinear in candidate pairs, ~n^1.5**:

| pairs | features + ILP |
|---|---|
| 8.7k | ~16 s |
| 55.6k | ~38 s |
| 73.6k | ~58 s |
| **272.6k** | **~383 s** |

3.7x the pairs costs 6.6x the time. `6bba_05db0fb1` is not even the
worst case the train set contains -- the densest 6bba movie runs 519
cells/frame. A handful of denser hidden volumes blows the budget.

Fixes, in the order they should be tried:

1. **Window the ILP in time.** One global solve costs ~n^1.5; W windows
   (overlapping by a frame so tracks are not cut) cost n^1.5 / sqrt(W),
   ~3.2x faster at W=10, and turn the superlinearity into near-linear so
   dense movies stop exploding. Needs a holdout re-validation (~30 min).
2. **Raise `ILP_FLOOR`** (0.10). At AUC 0.9884 true edges score high, so
   0.25-0.30 should cost nothing and cuts solver edges. Note it does NOT
   cut feature cost -- features are built for every in-gate pair and only
   then filtered by probability.
3. **A wall-clock guard, needed regardless.** Track elapsed time and
   remaining datasets; when the projection exceeds the cap, fall the rest
   back to the Hungarian linker that is already in the file. That floors
   the result at baseline behaviour and guarantees a valid CSV. Rule 1:
   print how many datasets fell back and from which one.

**User's call (2026-08-23), overriding the recommendation to fix first:**
submit anyway. Their reasoning, which is sound: a timeout does not burn a
submission slot, and slots are not the scarce resource -- producing one
real improvement takes days, so the account rarely spends even one slot a
day. 11.8 h against a 12 h cap may simply pass. The optimisation work
above stays queued either way.

## 2026-08-23 -- ⭐⭐⭐ div_w 0.55: 0.9363, and it beats the baseline on BOTH terms

`biohub-ranker-fine` filled the 0.20-0.75 gap the diag left. Eight points
now, one clean unimodal curve, argmax 0.55.

| div_w | ILP forks | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|
| 0.20 | 1024 | 0.9280 | 0.9193 | 0.0870 | 6/57/6 |
| 0.35 | 394 | 0.9309 | 0.9211 | 0.0980 | 5/39/7 |
| 0.45 | 190 | 0.9344 | 0.9216 | 0.1282 | 5/27/7 |
| 0.50 | 148 | 0.9345 | 0.9214 | 0.1316 | 5/26/7 |
| **0.55** | 111 | **0.9363** | **0.9220** | **0.1429** | 5/23/7 |
| 0.65 | 49 | 0.9351 | 0.9222 | 0.1290 | 4/19/8 |
| 0.75 | 12 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |
| 1.50 | 0 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |

**The win is not a trade.** At 0.55 both terms beat the xl baseline
(adj 0.9185 / div 0.1333): adj_J +0.0035 AND div_J +0.0096. Divisions
end up BETTER than the pipeline whose divisions the ranker was accused
of draining -- the capacity-2 solve more than repays what 1:1 stole.

Caveat to carry: the peak sits 0.0018 above its neighbours on SMALL
COUNTS -- div_J moves on FP 27/23/19 and TP 5/5/4 against 12 GT
divisions. The computation is deterministic (identical AUC 0.9884 in
all three training runs), so this is not run-to-run noise, but the
argmax is not sharply resolved: 0.45-0.65 are all defensible and 0.55
is simply the best measured. Predicted LB at the -0.015 family offset:
~0.921 vs the standing 0.917.

**Submission kernel built: `biohub-submit-ranker`** (`submit_ranker.py`).
Detection side is xl-repro verbatim; the linker is ranker probs -> the
capacity-2 ILP at div_w 0.55; every post-link patch is ranker-diag's,
because those -- not xl-repro's -- are the code path that measured
0.9363 (they differ: xl's gap-close sorts proposals by distance and
takes the closest under the cap, ranker's takes them in Hungarian
order). Verified before pushing:

- **Detector parity, the one that could have killed it.** The ranker
  trained on `biohub-edge-ranker`'s caches, which mount xiaoleilian's
  two weight datasets -- the same pair xl-repro uses -- with identical
  branches, flip views and POOL/CAND_THR/UNET_THRESH/NMS_UM/SCALE. The
  ranker learned on the xl detector's own coordinates, so the
  distribution shift that buried the chimera does not apply.
- **Offline.** Internet is off at submission; tracksdata/pyscipopt come
  from the support-pack wheels and `predict_unet_transformer` from the
  pack's repo copy. No clone anywhere.
- **An AST pass for undefined names caught two real graft bugs** that
  compile fine and die at runtime: `class UNet3D` and `_dist_um` were
  missing, because the block extractor took `def` and not `class`. Run
  that check on any assembled script.
- `submission.csv` is written BEFORE scoring, and scoring is wrapped
  whole: a scoring failure must never cost a submission. On the hidden
  rescore the train-GT lookup misses and scoring says so and skips.

## 2026-08-23 -- ⭐⭐ diag: the knob was live, the window was dead, and the reconciliation works

`biohub-ranker-diag` counted out-degree-2 nodes in the SOLVED graph
before any patch touched it, over a widened div_w sweep. This is the
measurement v4 could not produce.

| div_w | ILP forks | final forks | patch added | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|---|---|
| 0.20 | **1024** | 1423 | 399 | 0.9280 | 0.9193 | 0.0870 | 6/57/6 |
| **0.50** | **148** | 590 | 442 | **0.9345** | 0.9214 | **0.1316** | 5/26/7 |
| 0.75 | 12 | 461 | 449 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |
| 1.50 | **0** | 451 | 451 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |

Fork count runs 1024 -> 148 -> 12 -> 0, monotone across a huge range:
`division_weight` was always wired correctly. **v4's 0.75-0.95 window
sat entirely in the dead zone** (12 forks at 0.75, zero above), so its
three "identical" rows were one solved graph three times. The
byte-identity was a badly placed window, not a broken knob -- worth
recording, because the pessimistic reading (mechanism never engaged)
was wrong and would have shelved a live line.

**The reconciliation is confirmed.** At div_w 0.50 the ILP natively
takes 148 forks, div_J returns to near-baseline (0.1316 vs 0.1333) and
finds one MORE true division than baseline (5 TP vs 4), while adj_J
holds at 0.9214 (+0.0029 over baseline). Taking both daughters
natively is exactly what stops the stealing, as diagnosed.

Clean decomposition of the +0.0027 over baseline, two separate sources:

- **+0.0013 is the ranker's edges alone** -- at div_w 1.50 the ILP forks
  zero times, divisions come purely from `_add_safe_divisions`, and the
  pipeline still scores 0.9331 vs baseline 0.9318.
- **+0.0014 more is native forking**, 0.9331 -> 0.9345 at div_w 0.50.

0.20 over-forks: most TPs (6) but division FP 57 and adj_J falls to
0.9193. The curve is peaked between 0.20 and 0.75 with only those two
brackets measured, hence `biohub-ranker-fine` (0.35/0.45/0.55/0.65)
before the submission slot is spent.

Kernels this round, all in the repo: `ranker_v4.py`, `ranker_diag.py`,
`ranker_fine.py`. The diag/fine pair skips the A line via
`P_MIN_SWEEP = ()` (measured twice, identical).

## 2026-08-23 -- ranker v4: the gate is cleared by 0.0001, and the knob never moved

Two runs. The first ERRORed at 8.5 min: `P` was bound to the pipeline
module at import, then clobbered inside the module-level `p_min` loop by
`pairs, rows, P, C = feats[t]`, so the B line's `P.build_graph` hit a
numpy array. Pure name shadowing; the import is now `as PIPE`. Also
hardened: the B line mapped ILP endpoints with `remap[int(a)]`, which
KeyErrors on the `edge_endpoint_missing` case `link_replay` has warned
about since 2026-08-20 -- now dropped loudly as a lower bound (rule 1).
The script lived only in a dead session's scratchpad and on Kaggle; it
is now `ranker_v4.py` in the repo.

| config | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl baseline | 0.9318 | 0.9185 | **0.1333** | 4/18/8 |
| A line, p_min 0.30 | 0.9305 | 0.9222 | 0.0833 | -- |
| A line, p_min 0.35 | 0.9284 | 0.9200 | 0.0833 | -- |
| A line, p_min 0.40 | 0.9278 | 0.9195 | 0.0833 | -- |
| **B line, div_w 0.75** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |
| **B line, div_w 0.85** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |
| **B line, div_w 0.95** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |

Composite is exactly `adj_J + 0.1 * div_J` (verified on all seven rows).
That decomposes the A line's failure precisely: the ranker buys
**+0.0037 adj_J** and pays **-0.050 div_J = -0.0050**, net -0.0013. The
daughter-stealing diagnosis is confirmed as a mechanism -- and div_J
sits at 0.0833 for every p_min, so no threshold reaches it.

**The B line's result does not yet mean what it looks like.** Three
surcharges spanning 0.75-0.95 returned identical score, adj_J, div_J
AND TP/FP/FN. Identical to every digit means one solved graph, i.e.
`division_weight` had no effect. With `edge_weight = -1.0 *
edge_prob`, a fork is taken iff the second daughter's prob exceeds
div_w, so either every accepted fork scored > 0.95, or **no fork was
accepted at all** and all 3 TPs still come from the post-hoc
`_add_safe_divisions` patch. Supporting the latter: B is still BELOW
baseline on divisions (0.1071 vs 0.1333, 3 TP vs 4). It recovered part
of what the A line lost, not all. So the +0.0013 over baseline reads as
"the edge gain finally is not fully cancelled", not as "capacity-2
takes both daughters natively" -- which was the entire hypothesis.

Predicted LB at the -0.015 family offset: ~0.918 vs the standing 0.917.
Inside the noise. **The pre-registered rule is technically cleared and
substantively unsettled**; a submission slot is the scarce resource,
GPU quota is not, so `biohub-ranker-diag` counts forks in the solved
graph before any patch, over a div_w sweep widened to 0.2-1.5.

## Current state (2026-08-21)

### 2026-08-22 VERDICT: from-scratch retraining is closed; main line = post-processing + model swap/ensemble

Session 1 (4 full epochs, clean split, official recipe, 10.3 h) delivered
the learning curve: eval score (acc x recall on the 12-holdout) =
0.8980 / 0.9000 / 0.9157 / 0.9121 — one epoch buys 0.90, then the curve
crawls (+0.016 over three epochs, last epoch negative). Target: the
public weights' 0.9835 at epoch 402. Real epoch cost: **2.53 h** (not
2.1; smoke extrapolation was optimistic). 50 epochs = 127 h. The
400-epoch tail is real; parity from scratch is unaffordable. Appeal
path (not scheduled): a 3-epoch LR-schedule probe (3e-4 + cosine) —
constant lr 1e-4 may explain some of the crawl. Warm-start surgery
stays rejected (user: twisted melons; 0.764 as evidence). **Main line
is now B+C: the learned edge re-ranker + luxury inference (TTA,
ensemble, sub-voxel refinement) + adopting/ensembling the public
xiaoleilian detector.** Artifacts kept: session-1 checkpoint (clean
split, eval 0.9157) ferried to dataset `faustine649/biohub-train-ckpt`
— an ensemble-diversity candidate; the resumable trainer harness works
(4 epochs, per-epoch full checkpoints, zero incidents).

### 2026-08-22 post-processing round (details in entries below when logged)

- **Orphan patch v2 (0.917-derived strict filters: mutual-nearest sister,
  both children continue, divergence >= 2.25 um): local +0.0132** —
  v2 cap15 0.9401 vs v1 cap15 0.9348 (reproduced) vs cap0 0.9269, div
  TP/FP 2/13 -> 3/10. Higher caps degrade. Submission v2 in validation
  (`biohub-submit-orphan` v2); v1's local delta transferred 1:1 to LB.
- **Velocity-into-cost: line closed.** First-ever sweep of link_replay's
  velocity_adjust (24-cell grid, baseline 0.9387 reproduced): velocity
  only gains where disappearance is low (max +0.0041 at 0.35/2.5/0.25)
  and no cell beats the incumbent 0.40/3.5/0.0 — it substitutes for high
  disappearance cost rather than adding to it, and costs TP. Velocity
  survives only as a FEATURE for the learned edge re-ranker (the 0.915
  anatomy, recon addendum 2026-08-22b), which is the next post-processing
  build.
- Kernel-engineering trap for the toolchain file: importing numpy BEFORE
  a pip block that resolves dependencies can half-upgrade numpy on disk
  ("cannot import name '_center'") — do heavy imports after pip.

### 2026-08-22 Branch C (dual track) status

- **XL detector beats the official on the clean ruler**: recall 0.766 /
  prec 0.736 vs 0.650 / 0.637; dim-quartile 0.73 vs 0.52 (both models
  ZH-naive — a pure generalization duel). Its score distribution is
  compressed; 0.15 is its calibrated operating point.
- **Path 2 (verbatim 0.917 reproduction)**: validated on the visible 4
  (patches active, ~86 s/dataset) and SUBMITTED as ref 55690436 —
  expected ~0.917; also tests badge reproducibility.
- **Path 1 (chimera: XL detect -> official edge head -> our ILP + v2
  patch)**: v1 died of a double-downsample clamp (open_dataset returns
  already-downsampled shapes); v2 ran clean but LOST locally — adj_J
  0.9118 vs 0.9337, composite 0.9204 vs 0.9401. Caveat: the holdout is
  the official detector's HOME turf (possibly its training data), i.e.
  this filter is structurally biased against foreign detectors, plus the
  edge head faces a coordinate-distribution shift. Decision rule: if
  55690436 lands ~0.917 on the LB (an xl-family pipeline winning away
  from home), the chimera earns an LB slot despite the local reading;
  if the reproduction also flops, the xl family is suspect and the
  chimera is buried.

**Measurement doctrine (user's, adopted):** the only ruler is the LB —
strictly the private 71%, of which the public LB is the closest readable
proxy (+-0.03, 29%). Every local score (Zebrahub bench, holdout, val
losses) is a FILTER, not a ruler: it may only decide what earns a
submission slot, never declare improvement. "Better" is a word the LB
says.

**LB history (from the submissions API):** 0.681 (08-17, classical) ->
0.876 (08-19, ILP+A) -> 0.881 (08-20, thr 0.35 / disapp 2.5) -> 0.764
(08-21, fine-tune — REJECTED) -> 0.889 -> 0.898 (08-22, orphan patch
v1 -> v2) -> **0.917 (08-23, ref 55690436: verbatim reproduction of the
kunaldesale/xiaoleilian pipeline — the current best, badge reproduced
digit-for-digit)**. Gap to LB top-20 (0.935): 0.018.

**FAMILY-CALIBRATION DISCOVERY (2026-08-23):** local-to-LB offsets are
family-specific — official-weights family: local 0.9401 -> LB 0.898
(-0.042); xl family: local 0.9318 -> LB 0.917 (-0.015). Local rankings
INVERT on the LB. Hardest evidence yet that the holdout inflates the
official weights (memorization). Rule: absolute local scores must never
be compared across detector families; only within-family paired deltas
are valid. **Chimera buried** by this rule: within-family it trails the
repro by 0.011 (0.9204 vs 0.9318), predicted LB ~0.906 < 0.917 — the
official edge head is the weak link for foreign coordinates, no slot
spent. Main line: evolve the 0.917 xl pipeline with post-processing
upgrades, each a within-family paired delta against holdout baseline
0.9318 (its div line: div_J 0.1333, TP/FP/FN 4/18/8).

- **LB 0.876** (submitted 2026-08-19: official pipeline + pilkwang weights +
  `--use-ilp --ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5`,
  `det-threshold 0.99`). Public 0.916 notebook is the bar; its badge is
  pre-patch, so the honest gap is unknown but of order +0.04.
- **Local best on the 12-dataset `6bba` holdout: 0.9387** (cached replay,
  edge threshold 0.40, disappearance 3.5) vs 0.9269 for the submitted
  config. Local is optimistic vs LB by ~0.05; use it for paired deltas only.
- **Converged levers:** edge threshold x disappearance (done), det-threshold
  (done — not a lever; the detection ceiling is the model). **The open line
  is a better detector, trained on Zebrahub.**
- **Zebrahub crops extracted: 41 crops, 7.5 GB**, all five movies, in
  `biohub-zebrahub-extract` v4 output under `crops/` (entry below).
- **Clean detection ruler built and read**: public detector = recall 0.65 /
  prec 0.64 on Zebrahub dense GT (entry below). All training reads there.
- **Closed today:** sigmoid edge activation (best 0.9314 vs softmax's
  0.9387 on the holdout — no reason to pursue); NMS-merge division
  hypothesis (sisters are born ~10.6 um apart; entry below).
- **Fine-tune v3 trained** (augmented, two-domain): ZH val loss 1.307 ->
  0.194, but competition-domain val loss 0.018 -> 0.047 — the drift guard
  fired. Weights saved; verdict pends the clean-ruler re-bench.
- **Orphan forks v3: div_J moved for the first time** — cap 15 -> 0.080
  (2 TP / 13 FP). adj_J came out NaN (n_total read at the wrong nesting;
  it lives at gt["extra"]), so the net edge-cost verdict pends the
  rescore.
- **Fine-tuned weights passed the filter**: ZH bench recall@0.99
  0.650 -> 0.726, precision 0.637 -> 0.729 (FP -20.7k), dim-quartile
  recall 0.52 -> 0.71. Caveat: the fine-tune TRAINED on these crops, so
  the ZH ruler is no longer clean for this model — it only certifies
  "learned, not broken". The real verdict is the LB.
- **Fine-tune v3 REJECTED by the LB: 0.764** (ref 55670839, -0.112 vs
  0.876, single-variable). The comp-domain drift warning was real and
  decisive; every local gain (ZH recall +0.075) was paid for out of the
  test domain. Next fine-tune iteration needs a stronger anchor: higher
  comp fraction / lower LR / fewer steps, or post-hoc weight
  interpolation toward the original — and the LB stays the judge.
- **Orphan forks: NET +0.0080 at cap 15** (official metric, holdout):
  cap0 reproduces 0.9269 digit-for-digit (harness validated); cap 15 ->
  0.9348, div_J 0 -> 0.080, and adj_J pays only 0.0001 — the added edges
  are effectively free, as constructed. cap 40 is worse (0.9332).
- **Campaign session 1 LAUNCHED** (`biohub-train-campaign` v1): official
  recipe, clean split, 4 epochs (~8.5 h), resumable (full checkpoint per
  epoch; ferry `last_full.pth` via a ckpt dataset between sessions).
  Measured cost: **~2.1 h/epoch on T4 x2 at any batch size** (GPU-bound,
  ~2.4 windows/s; bs 8 hit a NaN anomaly — avoided; expandable_segments
  cures the bs-16 OOM). Budget: 30 h/week, full tank, ~180 h to deadline;
  no Kaggle paid tier (the one upgrade path: Colab Pro unlocks extra
  Kaggle GPU hours). CLI trap logged in kaggle-toolchain.md: access
  tokens expire and the CLI won't auto-refresh — inject
  `KAGGLE_API_TOKEN` from `kaggle auth print-access-token` per call.
- **De-risking doctrine for expensive training (user's concern: no
  one-shot 100 h bets):** (1) session 1 buys the LEARNING CURVE — how
  many epochs matter; (2) design variants go through a SCREENING round
  first (max_iters-sampled epochs, ~3 h/variant, ranked on ZH bench +
  holdout); (3) loss/data/augment changes branch WARM-STARTED from a
  shared trunk checkpoint — only architecture changes force from-scratch;
  (4) exactly one full-scale run, of the screened composite. AMP (absent
  from the official script) to be validated before screening — likely
  halves every number above. (An earlier draft here cited "pilkwang's
  50ep pack proves a 50-epoch Kaggle run" — struck; see the epoch-402
  correction below.)
- **STRATEGIC PIVOT (user's call, 2026-08-22): the training campaign is
  the main line.** Knob-tuning is provably exhausted (three days bought
  +0.013; the 0.952 author's sweep log is flat), the +0.046 gap to the
  top 20 lives in the edge term, which is model-bound. Plan: (0) smoke-run
  the official recipe for real epoch timing + first clean-split checkpoint
  — `biohub-train-smoke` in flight; (1) short reproduce on competition
  data, our split; (2) two-domain FROM-SCRATCH train (no catastrophic-
  forgetting frame): package Zebrahub crops as competition-format
  zarr+geff fakes (must carry `image_statistics.quantiles` in attrs — the
  loader raises without them) and add them to a custom splits file — zero
  changes to official training code; (3) train the edge head on
  Zebrahub's 20M edges. **CORRECTION (user's, 2026-08-22): the working
  public weights carry `epoch: 402` in their checkpoint — they are a
  ~400-epoch product, NOT 50; the pack's "50ep" name proves nothing about
  the edge predictor's training length.** At our measured 2.1 h/epoch,
  400 epochs ≈ 840 h — from-scratch parity may be unaffordable. Budget
  consequences: (a) the session-1 learning curve decides whether
  from-scratch is live at all; (b) the default trunk for improvements
  becomes CONTINUED training from the public weights (original loss,
  original data plus additions, low LR — unlike the rejected fine-tune,
  which changed loss weights and fed 70% foreign data); (c) AMP and
  max_iters sampling graduate from nice-to-have to required. Stage-2 ingredient (user's diagnosis, 2026-08-22):
  **the loss never supervises cell COUNT** — sparse dots + neg weight 0.01
  means counting is a training blind spot; that is why recall collapses
  with density (corr -0.68, ZH 0.65), why N/N_true drifts 1.01-1.30
  uncalibrated, and why ILP disappearance (a crude post-hoc counter) was
  the biggest post-processing lever. Fix: a count-consistency loss —
  heatmap probability mass pulled toward the known count (N_true exists
  for all 199 train sets; Zebrahub crops have exact per-frame counts),
  plus raised neg weight on dense crops; `cell_count.py` per-dataset
  estimates remain the inference-side fallback (+0.155 evidence in the
  classical era). Engineering gap: the official trainer has NO
  resume (best-weights only, no optimizer state) — 12 h kernel cap means
  a checkpoint-chain harness. Facts: trainer is Kaggle-native (T4 x2
  DataParallel hint in-code; pilkwang's "50ep" pack proves a 50-epoch run
  happened on Kaggle). Filters stay filters; every stage's verdict is
  the LB.
- **LB 0.889 — NEW BEST** (ref 55674219): the 0.881 config + the cap-15
  orphan patch, ported to a no-GT in-memory pass (`dr._candidates_one` on
  the solved graph, image feature via the zarr-free reader). The
  division term finally earns on the LB. Follow-ups on this line: tune
  the cap (15 was picked from a 4-point sweep), add the source-side
  mass-balance feature (the salvaged softmax idea), and re-measure the
  patch on top of any future linking config.
- Uncommitted: this docs restructure, `zebrahub_extract.py` rewrite.

### To do

- [x] ~~Measure the public detector on the 41 Zebrahub crops~~ — done,
      **recall 0.65 / precision 0.64** at the pipeline's det gate
      (entry below). The ruler works, has huge dynamic range, and costs
      6 min per full pass (`biohub-zh-detect-bench`).
- [ ] Fine-tune on the crops, re-run the bench, and if the gain is large,
      package weights as a dataset and SUBMIT — the LB (resolution ~0.03)
      is the final arbiter, per the user's call.
- [ ] Wire the 41 crops into `finetune_detect.py` (per-crop quantiles;
      dense labels need a different neg-weight scheme than the sparse-GT
      0.1; treat `parent_track_id < 0` OR the int64-cast NaN garbage as
      "no parent" — see the harvest entry).
- [ ] **Explain the 0.051 local-vs-LB offset.** Live probe: a `44b6`-only
      holdout — every number we own is `6bba`, which has a 7.47%
      duplicate-frame rate vs `44b6`'s zero. One inference pass.
- [ ] **Field-of-view drift** [CPU, GT only]: per frame pair, median
      displacement VECTOR of GT cells; coherent drift vs cancelling motion.
      Decides whether population growth can estimate division counts, and
      feeds appearance-based division recovery.
- [ ] `--ilp-division-weight` between 1.0 and 0.3 (0.85 would fork only on
      very confident second edges). Untested.
- [ ] `6bba_7af54fde`: 97 cells/frame yet J=0.828, worst over-detection
      (1.28). Motion suspected; one `run_gif` + displacement percentiles.
- [ ] Deferred: `cell_count.py` on real data (only if N_true estimation
      helps the node-count penalty).

Done/dead, for the record: ILP+A confirmed 12/12 (2026-08-19, 0.9269);
submitted (0.876); runtime budget resolved (5.2 h of 12, see reference);
det-threshold sweep (2026-08-21, dead as a lever); re-running the 0.916
notebook (dropped — pre-patch badge measures the closed exploit).

---

## 2026-08-21 — sigmoid edge activation: no win; sisters born far apart

**Sigmoid recache** (`biohub-cache-sigmoid`: edge_activation=sigmoid at
dump threshold 0.05, then a 7x3 replay grid). Mechanism confirmed: with
sigmoid, candidates above 0.5 EXCEED the target count (impossible under
softmax) — the mass-splitting starvation is real and softmax-specific.
But the score does not follow: grid best 0.9314 at threshold 0.70 /
disapp 3.5 vs softmax's 0.9387 at 0.40/3.5; edge_J lower everywhere
(0.9316 max vs 0.9363). The freed candidates are mostly the spurious pool
the stranded audit predicted; the ILP buys FPs with them. At threshold
0.9 the sigmoid graph collapses outright (adj_J 0.05 — probabilities
rarely reach 0.9). Verdict: line closed. (Holdout-measured, so the
negative carries the usual trust discount — but nothing here motivates
spending more on it.)

**A salvageable idea from the sigmoid post-mortem** (user's): the official
softmax normalises per TARGET (candidate parents compete), so a division
does NOT produce a 0.5/0.5 signature — each daughter's column gives the
shared parent ~1.0. But the transposed statistic can be built from the
cache: renormalise each SOURCE's outgoing candidate probs; a parent whose
top-2 outgoing shares are balanced AND whose edge lengths sit in the
division band (3.5-11 um) is a synthetic fork signature. Pure CPU, cache
only — a third feature for the orphan-fork scorer (existing: edge length
AUC 0.86, parent brightness AUC 0.71) if the base pass proves out.

**Sister-separation curve** (`biohub-sister-curve`, all 151 GT
divisions): at k=0, the daughters' first frame, separation p10/p50/p90 =
6.4 / 10.6 / 14.4 um; only 0.7% closer than the 3 um NMS kernel. The
2026-08-19 "NMS merges newborn daughters" hypothesis is dead: daughters
are annotated far apart from birth. Late-reporting within the +-1-frame
window solves nothing because nothing is merged. Division losses must
come from elsewhere (detection/linking of the second daughter — exactly
what the orphan-fork pass targets).

**Two kernel-engineering lessons** (cost: ~30 wasted GPU-min): (1) the
pack renames `tracking_cellmot` -> `biohub_tracking` AND its
`evaluate.py` is a different version without `evaluate_pairs`;
`division_recovery.sweep`'s import of it had never been executed. Score
against the official metrics API directly. (2) An ERRORed kernel's output
files are downloadable but CANNOT be attached as a kernel_source — a
run that dies at the last step strands its artefacts. (3) Loading all 41
crops decompressed needs ~33 GB RAM — OOM-killed finetune v2; use
/kaggle/tmp memmaps.

## 2026-08-21 — the clean ruler's first reading: recall 0.65, not 0.99

`biohub-zh-detect-bench` (v2): public weights, untouched, on all 41
Zebrahub crops — ~206k dense GT points, every 5th frame, greedy 1:1 match
within 5 um, NMS pool 3 um, no TTA. 6 minutes of T4.

| det gate | TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| 0.5 | 148,656 | 99,760 | 57,304 | 0.722 | 0.598 |
| 0.9 | 143,621 | 89,911 | 62,339 | 0.697 | 0.615 |
| **0.99 (pipeline)** | 133,787 | 76,243 | 72,173 | **0.650** | 0.637 |

Recall@0.99 by GT-point brightness quartile: Q1 (dimmest) 0.52 — partly
Zebrahub label noise; **Q4 (brightest) 0.74** — those are unambiguous real
cells, and the detector misses a quarter of them. Label noise cannot
explain the gap; the holdout's node_recall 0.9974 cannot survive it
either. Per-crop spread 0.32–0.92 (`zh_detect_bench.csv` in the kernel
output); ZSNS004 (a dim movie) is worst, dense ZSNS003/005 crops are fine.

Reading it honestly, three causes stack and this run cannot separate them:
(1) the holdout number is inflated (memorization + duplicate frames);
(2) genuine domain shift — Zebrahub is a different instrument setup, and
the hidden test (a third embryo on the COMPETITION instrument) sits closer
to the holdout than to Zebrahub, so 0.65 is a lower bound story, not the
hidden-test number; (3) method deltas (no TTA, approximate NMS) — worth a
few points at most.

What it establishes regardless: the ruler has dynamic range (0.65 baseline,
0.35 of visible headroom), a real fine-tune effect will be unmissable on
it, and "detection is solved" is dead as a general claim — it was only
ever measured where the model may have been grading its own homework.

**Correction (user's, accepted): "the hidden test is on the competition
instrument" was asserted without evidence — and (second push) so was
"same format".** Nobody has seen the hidden data; the visible test/ files
are train copies. And (third push, also the user's) "the pipeline read it
fine" proves NOTHING about similarity — Zebrahub itself, a different
instrument outright, goes through the same pipeline after resampling, as
this very bench demonstrates. Format compatibility and runtime only pin
format family and data volume. **The single load-bearing fact about
hidden-test similarity: train-tuned pipelines can reach 0.93+ on it —
LB #1 0.957 / #20 0.935, read via API 2026-08-20, post-rescore; our own
bare config 0.876.** (NOT "the whole field scores 0.9+" — an earlier
draft of this sentence overclaimed; most of the 2560 teams score below
0.9.) That top-end existence proof bounds the domain shift to an
eroding-percentage-points factor, not a new world; the public LB is 29%
of the test, so the bound carries that caveat too. Everything beyond
this is conjecture.
Photometrics are free to differ per batch — the competition's own two
batches differ 8x in brightness. Evidence the detector is photometrically brittle WITHIN one
instrument: ZSNS004 (dim) is the worst movie on this bench (crops down to
0.32 recall). So part of the local-vs-LB offset may be domain shift we
have never mitigated. And (user's addition) **brightness is only the
observed axis — sharpness/PSF width, noise character, scattering with
depth, and axes we have not thought to look at can all differ per batch
and are so far unmeasured.** Cheap probe if wanted: per-dataset sharpness
proxy (e.g. Laplacian energy / gradient p95 on a mid frame) across 44b6 /
6bba / Zebrahub, same style as the brightness survey. Consequences for
the fine-tune: (1) augmentation must cover more than gain — random
gain/offset/gamma/noise AND Gaussian blur of varying width (a blur-robust
detector is insurance against a softer-imaging test set); (2) mix
competition sparse-GT frames with Zebrahub dense crops — features that
hold across two domains stand a better chance on a third. LB submission
remains the only direct read on the hidden domain.

## 2026-08-21 — validation trust: null results on the holdout are void

User's argument, accepted: the 12-dataset `6bba` holdout is duplicate-
frame-inflated AND possibly inside the public weights' training split, so
a measurement of "no improvement" there is uninterpretable — a memorized
baseline leaves no visible headroom even for real gains. Corollary: if
0.982 detect_ceiling is memorization, the hidden-test detection is worse
than the books say and detector training is worth MORE, not less.

Trust tiers, applied from now on:

- **Still valid:** LB 0.876 (real); the scorer cross-check (same-graph
  comparison); the softmax-gate mechanism (read from code, not from the
  holdout); all Zebrahub facts (independently verified).
- **Discounted:** every absolute number on the holdout (0.9269, 0.9387,
  detect_ceiling 0.9823, node_recall 0.9974).
- **Void:** every NULL result measured on the holdout — including
  "det-threshold is not the lever" (the direction may still be dead, but
  the evidence is weak) and any "no effect" from the in-flight sigmoid /
  orphan-fork runs. Positive paired deltas keep some value; zeros do not.

Second, independent problem (also the user's): **no statistical power at
saturation.** The holdout has ~10.6k GT edges and ~180 detection-missed
endpoints; fixing 10% of the miss pool moves edge_J by ~0.0017 — the same
order as run-to-run noise (seeds, ILP tie-breaking, TTA), whose floor has
never even been measured. In the 0.98 regime, signal and noise are the
same size; small deltas in either direction prove nothing.

The fix for both: the 41 Zebrahub crops double as the first
contamination-proof detection benchmark (dense GT, ~900k nodes = ~100x
the holdout's effective sample; weights trained only on competition data
per the official training scripts). Measure the current detector there
before spending GPU on training; read all training experiments on that
ruler, where a real effect has room to be visibly larger than noise.

## 2026-08-21 — extraction v2 harvested: 41 crops, 7.5 GB, zero empty

`biohub-zebrahub-extract` v4 (SCRIPT_VERSION 2026-08-21d) ran 94 min and
completed cleanly. Per movie (units all detected correctly; crops with
cells/frame at T0):

| movie | units | T0 | crops | cells/frame span | divisions in full CSV |
|---|---|---|---|---|---|
| ZSNS001 | UM | 237 | 7 | 51-541 | 635,042 |
| ZSNS001_tail | UM | 237 | 5 | 63-566 | 36,878 |
| ZSNS003 | VOXEL | 154 | 9 | 43-384 | 15,869 |
| ZSNS004 | VOXEL | 180 | 10 | 52-578 | 30,279 |
| ZSNS005 | VOXEL | 180 | 10 | 36-672 | 46,027 |

- Density now spans 36-672 cells/frame, bracketing the competition's
  50-519, with sparse crops included — the v1 failure (3 empty, 2
  over-dense) is fixed. _tail's slab lacked the 80-250 bins (loudly
  degraded, accepted).
- Displacements p50 1.32-1.58 um everywhere; ZSNS004 p99 10.13 um is
  tail-heavy vs the competition's 6.71 — fine for detection, watch it if
  training a linker on 004.
- **Defect:** `ZSNS001_tail` uses NaN (not -1) for "no parent";
  `astype(np.int64)` cast it to garbage (RuntimeWarning in the log). All
  five tail crops carry it. Consumers must treat `parent_track_id < 0` as
  no-parent; the next extract revision should `fillna(-1)` before the
  cast. Division counting via "parent with >=2 children" must also drop
  the garbage id or every root pair looks like one giant fake division.
- Total division supervision now on disk: ~764k events across five movies
  vs the competition's 151.

## 2026-08-21 — extraction v2 (all five movies, density-bin placement)

`zebrahub_extract.py` rewritten (SCRIPT_VERSION 2026-08-21d), pushed as
`biohub-zebrahub-extract` v4. Changes, each answering a measured failure:

1. All five movies in one run, per-movie try/except, 14 GB output budget.
2. Units DETECTED per movie (coordinate max vs physical extent), columns
   normalised — the two traps in `reference.md#zebrahub-external-data`.
3. Placement: up to 2 crops per density bin {30-80, 80-150, 150-250,
   250-400, 400-650 cells/frame} inside the best z-slab, replacing "densest
   3x3 tile" which produced 3 empty crops and 2 over-dense ones (v1
   harvest, below). Small-chunk movies (001, _tail) confine candidates to a
   4x4-bin window so the per-frame read stays bounded; whole-plane-chunk
   movies (003/004/005) pay for the plane anyway and are unconstrained.
4. Fixed a silent whole-frame read: zarr 2 materialises `arr[t, 0]` (the
   full 4.7 GB frame) before a second `[slices]` applies. v1 downloaded
   every full frame without knowing. One-call indexing now (trap 22).

Placement logic dry-run locally on sampled real data before pushing:
4 crops covering 3 bins on the 7%-sample (bins fill at full density), bbox
108 MB raw/frame, within budget.

## 2026-08-21 — competition brightness is not uniform either

Question raised by the user: is per-dataset normalisation throwing away a
real feature, and is the competition's own brightness uniform? Measured
(`faustine649/competition-brightness-survey`): q0.999 spans 8.8x across the
199 train datasets — same regime as Zebrahub's 10x. The organizers store
per-dataset quantiles in the zarr attrs and the official `io.py` normalises
with them, so per-dataset normalisation is the designed-in policy, not a
choice we are making. Numbers and the frame-vs-movie caveat:
`reference.md#train-set-survey-199-datasets`.

Absolute brightness is instrument+expression+bleaching, not biology;
a linear per-dataset rescale preserves all within-image contrast and
ordering. `cell_count.py` had already excluded absolute-intensity features
for exactly this reason (kept only as a LOGO control).

## 2026-08-21 — first extraction run harvested (v1, ZSNS001 only)

`biohub-zebrahub-extract` SCRIPT_VERSION 2026-08-21a completed ~11:35 and
sat unread until evening (lesson: COMPLETE is not harvested). Results:

- Full CSV: 21,697,591 nodes, 635,042 divisions; displacement p50 1.24 /
  p99 5.19 um at t=200 — compatible with the competition's 1.82 / 6.71.
- 9 crops, 1278 MB, `crops/*.npz` in the kernel output. Six labelled
  (159 / 523 / 531 / 552 / 1041 / 1211 nodes per frame), **three empty**
  (densest tile hangs off the embryo edge — predicted by the pre-read).
- Two crops (1041, 1211/frame) are denser than anything in the competition
  (true 50-519/frame). "Maximise density" was the wrong placement
  objective -> replaced in v2 (entry above).

## 2026-08-21 — Zebrahub verified as training data: all 5 movies PASS

Re-evaluated the source bucket from the data alone, trusting no prior
assertion (user's instruction). Local probe on ZSNS001 (60 MB CSV head +
one zarr chunk), then `faustine649/zebrahub-align-spotcheck` ran four
checks per movie: units, alignment (the test the `kkunizaw` mirror failed),
displacement, density. All five PASS; full fact table and the traps now
live in `reference.md#zebrahub-external-data`.

What the check caught that assertions missed:

1. ZSNS003/004/005 coordinates are **voxel indices, not um** — the naive um
   reading fails alignment outright (v1 of the spot-check FLAGged both and
   crashed on 005; unit detection turned them into PASSes).
2. `ZSNS001_tail` renames columns (`ParentTrackID` etc.).
3. First displacement measurement said p50 = 0.00 — a self-inflicted bug
   (diffing across sampling gaps instead of dt==1 only). Corrected: 2.9%
   exact zeros, p50 1.52.

Verdict: suitable for training. Caveats that survive: machine-generated
labels (~10% near background — weight by local brightness), per-crop
normalisation mandatory, ZSNS002 unusable.

## 2026-08-21 — det-threshold is not the lever; the ceiling is the model

Scored on the cache at the converged linking optimum (threshold 0.40,
disappearance 3.5), so only `det_threshold` varies:

| det | adj_J | edge_J | detect_ceiling | link_rate | nodes |
|---|---|---|---|---|---|
| **0.99** | **0.9387** | 0.9363 | 0.9825 | 0.9839 | 161,649 |
| 0.98 | 0.9375 | 0.9362 | 0.9827 | 0.9836 | 164,185 |
| 0.95 | 0.9367 | 0.9365 | 0.9821 | 0.9842 | 166,209 |

`detect_ceiling` does not move and neither does `edge_J`; the adj_J fall is
entirely the node-count penalty on 2.8% more nodes. The extra detections
admitted by a lower gate are NOT the missing endpoints of the ~180
unreachable GT edges — those cells are not peaks in the heatmap at any
threshold down to 0.95. **The detection ceiling is set by the model's
output, not by our threshold.**

Pattern worth keeping (three wrong directions argued on this one knob, two
from stories about model internals, one sound-but-insufficient from the
metric alone): **score the thing.** The deflation story was wrong on
arithmetic: the negative loss term weighs ~2.4e-8 per unannotated voxel.

What it leaves open: a better detector, which means training — Zebrahub
(entries above).

---

## 2026-08-20 — the candidate cache, and what it measured

GPU inference now runs once (`cache_candidates.py` dumps coords/edges/probs
at threshold 0.05); every linking experiment replays on CPU
(`link_replay.py`, Kaggle entry `run_sweep.py`). One 12-dataset pass:
154 s CPU vs 11 min GPU. Cache verified faithful: replaying the submitted
config reproduces TP/FP/FN 9863/318/450 and adj_J 0.9269 digit for digit.

### Correction: 0.9953/0.9530 was the GREEDY decomposition

| | detect_ceiling | link_rate |
|---|---|---|
| as previously recorded | 0.9953 | 0.9530 |
| ILP+A, measured | **0.9823** | **0.9736** |

ILP wins by giving up detection reach (solver discards nodes) to buy
linking. Detection's share of the loss is 1.8%, not 0.5% — still
linking-dominated, but 3.5x what the log claimed.

### 11.8% of targets have no candidate parent at all

Pooled over the holdout at dump threshold 0.05: candidates above 0.5 =
167,398 (everything the submission ever saw); above 0.05 = 338,781 (2.02x);
targets = 189,872, of which **22,474 (11.8%) have no candidate above 0.5**
— structural false negatives. The softmax normalises over sources, so at
most one parent per target can exceed 0.5.

Stranded fraction correlates with over-detection (`N/N_true` r=+0.79), NOT
density (r=-0.07). Earlier "density is the primary failure axis" holds for
the score, not for this mechanism: a spurious detection has no true parent,
its column mass spreads, nothing clears 0.5.

### Counted, not estimated (`stranded_audit.py`)

1. Stranded targets are disproportionately spurious: GT-match rate 1.5% vs
   6.1% for confident targets (ratio 0.24, holds in all 12 datasets). Most
   of the 22,474 should stay unlinked.
2. Yet **278 of the 450 edge FNs (62%) are real edges stuck behind the 0.5
   gate** (27 more have an undetected target). Recovering all 278 at F new
   FPs: +0.026 at F=0, break-even near F=300, and the pool holds 22,196
   spurious targets — the real edges are 1.2% of it. Needle in haystack;
   whether it pays is entirely ILP selectivity.
3. Prediction for the 2-D sweep: the winning cell pairs a lower threshold
   with a LOWER disappearance weight (the current bias towards linking
   becomes a liability at a lower gate).

Over-detection became the higher-leverage target here — it pays the
node-count penalty AND buries the 278 recoverable edges.
`SUPERSEDED 2026-08-21`: the det-threshold knob was then swept and is dead
(entry above); the over-detection problem still stands, but its fix is a
better detector, not the threshold.

## 2026-08-20 — ⭐⭐ the edge threshold is the binding constraint

Read verbatim from `scripts/predict_unet_transformer.py` (defaults now in
`reference.md#running-the-official-pipeline`): the edge-candidate gate
`threshold=0.5` is the only limit on what the linker sees, and the softmax
normalises over sources. A target whose column mass splits between two
plausible parents (~0.5/0.5) gets NO candidate edge at all. This predicted,
before measurement: the 60% `DROPPED` failure bucket, `corr(edge J, true
density) = -0.68`, the densest movie being worst, and detect_ceiling >>
link_rate.

Arithmetic headroom estimated then: up to +0.041 if every dropped edge
recovers and retires an FP. `SUPERSEDED 2026-08-20` (same day, audit above):
the stranded pool is mostly spurious, so +0.041 was optimistic; the counted
recoverable number is 278 edges, +0.026 max at zero cost.

Consequences that stand:

- Sweep `threshold` x `disappearance_weight` TOGETHER (parameters
  interact; appearance 0.0 / disappearance 1.5 currently biases towards
  linking).
- Forks stay impossible throughout (needs edge_prob > 1.0), so the sweep is
  safe for the division term.
- `sweep_edge_threshold.py`'s greedy-only monotonicity argument does not
  transfer to ILP — hence the candidate cache instead of geff replays.
- The sweep landed at threshold 0.40 / disappearance 3.5 -> local 0.9387
  (state block). [Recorded 2026-08-21 alongside the det sweep.]

---

## 2026-08-19 — submitted: LB 0.876, local 0.9269

Official repo + pilkwang's public weights + two flags (`--use-ilp`,
appearance 0.0 / disappearance 1.5), `det-threshold 0.99`. No DeepCenter,
no second seed, none of the ~40 `BIOHUB_*` knobs. Up from 0.681 (classical
DoG pipeline).

| | LB | local (12 held-out `6bba`) |
|---|---|---|
| classical DoG | 0.681 | ~0.45 |
| **official + ILP+A** | **0.876** | 0.9269 |
| public 0.916 notebook (full stack) | 0.916 | never measured post-patch |

### The 0.051 offset: local is optimistic

Candidate causes, none separated yet: (1) the primary U-Net's split is
unknown — if it trained on the holdout, 0.9269 is inflated; (2) the hidden
test may be denser/differently distributed (density drives the score,
r=-0.68); (3) the node-count penalty depends on the density mix.
Operationally: **use local scores for paired deltas only** — "local 0.93 so
we are near 0.93" is false.

### Local best detail (12 datasets, official `evaluate.py`)

```
score              0.9269   (greedy default: 0.8847, improved 12 of 12)
edge_jaccard       0.9278   (TP/FP/FN 9863 / 318 / 450)
division_jaccard   0.0000   (0 forks emitted; 12 GT divisions all FN)
node_recall        0.9882
node-count penalty 0.0009   (was 0.0140)
```

Gain decomposition greedy -> ILP+A: edge_J +0.0318, penalty +0.0131,
division term -0.0027. TP +80, FP -288 (-48%), n_pred -12%.

Per-dataset (sorted by true density; `N/N_true` from geff
`estimated_number_of_nodes`):

| dataset | edge J | true cells/frame | N/N_true | div FP (greedy) |
|---|---|---|---|---|
| 6bba_3abfe10a | **0.7705** | **519** | 1.21 | 42 |
| 6bba_5c824876 | 0.9033 | 246 | 1.13 | 29 |
| 6bba_283bf9f1 | 0.9171 | 209 | 1.12 | 28 |
| 6bba_2312ac41 | 0.9611 | 119 | 1.14 | 11 |
| 6bba_61dd1e0d | 0.9164 | 102 | 1.09 | 14 |
| 6bba_7af54fde | **0.8280** | 97 | **1.28** | 29 |
| 6bba_3a1849c2 | 0.8783 | 70 | 1.17 | 20 |
| 6bba_c27cba08 | 0.9107 | 64 | 1.12 | 5 |
| 6bba_afb141ff | 0.9070 | 60 | 1.17 | 5 |
| 6bba_7b5d3b2c | 0.9596 | 58 | 1.01 | 8 |
| 6bba_268e1230 | 0.9016 | 51 | 1.30 | 15 |
| 6bba_aeee7805 | 0.9680 | 50 | 1.09 | 8 |

`corr(edge J, true density) = -0.68`, `corr(edge J, N/N_true) = -0.64`.
Two failure modes, separately attackable: crowding, and fast movement
(`7af54fde` is slow-density but fast-motion; the `6bba` batch's median
displacement p99 of 7.81 um exceeds the 7 um match radius).

### Division accounting at this config

Greedy emitted division TP/FP/FN 6/214/6 (div_J 0.0265); ILP+A emits no
forks: 0/0/12, ceiling 1.0 worth +0.1, realistic target the orphan-fork
pass's measured +0.0129. `SUPERSEDED-CORRECTION 2026-08-20`: an earlier
"18x fork over-prediction" claim was circular (220 = 6+214, never counted
from the graph); the metric only sees matched forks, so the true
over-prediction factor is unknown. Fork-leniency details:
`reference.md#the-metric`.

Headroom table (greedy config): div FP 214 -> 0 buys +0.047; past that
needs recovering the 6 FNs.

### Local scorer verified against official

Same 12 predicted graphs: edge TP/FP/FN identical (9783/606/530), adj_J
agrees to 5 decimals; division term differs slightly (ours 231 FP vs
official 214 — lenient approximation, quote the official). Licence to use
the fast local scorer for paired sweeps. `edge_fp_lenient = 5`: the old
scorer saw five of the 606 FPs.

### Divisions: negative feedback outweighs positive at the tested setting

`--ilp-division-weight 0.3` (scope: one knob, one notch, 3 datasets, 4 GT
divisions): div_J 0 -> 0.0308 (+0.0031) but adj_J 0.9152 -> 0.8907
(-0.0245). 8:1 against; default stays 1.0 (zero forks). Also measured:
missed divisions corrupt the edge term (GT edges leaving a dividing cell
fail at 29.2% vs 5.1% baseline) — worth roughly +0.001, real but small.
Obstacles: 151 divisions total (sample size); NMS `pool_kernel_um` may
merge daughters for ~3 frames — UNTESTED (`division_gif_cell.py` settles it
in a minute). Untried: weights between 0.3 and 1.0, smaller pool kernel,
deliberately-late splits within the ±1-frame window, official
`division_metrics.py` locally.

---

## 2026-08-18 — ⭐ ILP linking: +0.0295 over greedy (3-dataset pilot)

Paired, same weights, det 0.99, on `3abfe10a`/`2312ac41`/`aeee7805`:

| config | edge J | adj_J | penalty | score |
|---|---|---|---|---|
| greedy | 0.8956 | 0.8826 | 0.0130 | 0.8857 |
| `--use-ilp` | 0.9150 | 0.9073 | 0.0077 | 0.9073 |
| **+ app 0.0, disapp 1.5** | 0.9183 | **0.9152** | 0.0031 | **0.9152** |
| + division-weight 0.3 | 0.8999 | 0.8907 | 0.0092 | 0.8937 |
| + disapp 3.0 | 0.9097 | 0.9120 | -0.0023 | 0.9120 |

Most of the gain is the node-count penalty, not edge J: high disappearance
cost makes the solver prefer long tracks and discard scattered nodes
(n_pred -12%). At disapp 3.0 the penalty goes NEGATIVE (n_pred < n_true;
the formula has no upper clamp) but node_recall pays more. Optimum near
1.5, curve flat.

## 2026-08-18 — ⭐ the scorer bug

Real rule (`tracking_cellmot/metrics.py::_evaluate_matched_graph`): an edge
counts toward the denominator iff **either** endpoint matched a GT node of
the right degree. Ours required BOTH. Measured on 24 classical-pipeline
datasets: FP 2 -> 289; 99.3% of false positives were invisible. Also
implemented from the real scorer: t+1-only edges, merge collapse,
out-degree cap 2. Full metric rules: `reference.md#the-metric`.

Consequence: a missed detection costs twice (FN + induced FP), so detection
recall is worth far more than pre-fix measurements implied — exactly why
the official U-Net (recall 0.997) beats DoG (0.70) by more than recall
alone suggests.

**Every local number from before 2026-08-18 is void** (produced by the
broken scorer). In particular the old "0.916 pipeline scores 0.9614
locally" is dead.

Re-measured on the classical pipeline with the corrected scorer:
over-detection still wins (adj_J rising through ratio 2.8 at N_pred/N_true
0.94 — the DoG detector runs out of candidates; `nms_um` is its real
lever).

## 2026-08-18 — published artefacts read

DeepCenter's split, its two checkpoints, and the primary model's unknown
split — moved to `reference.md#published-artefact-facts-read-2026-08-18`.
Actionable consequence: the local holdout is `6bba`-only because DeepCenter
provably never saw `6bba`.
