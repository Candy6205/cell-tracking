# Progress log — Biohub Cell Tracking During Development

> Reader: a future AI session. Strictly reverse-chronological; the newest
> entry and the state block below are always at the top. Stable lookup facts
> (metric, survey, mechanics, code map, traps) live in `reference.md` — this
> file records events and measurements only. When a later entry overturns an
> earlier claim, the earlier text carries a `SUPERSEDED` mark in place.
> Engineering conventions: `engineering-rules.md` (rule 1: degradation is
> never silent).

## HANDOFF (2026-08-24, written at the end of a very long session)

**LB 0.921, rank ~155 / 2658.** The best submission is **ref 55707206**:
xl detect + learned edge re-ranker + capacity-2 tracksdata ILP at
div_w 0.55, **solved GLOBALLY (no windowing)**.

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

## ⭐ PRIORITY as of 2026-08-24 (user's call): detection and linking, not divisions

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

## 2026-08-25 -- ⭐⭐⭐ THE RANKER TRAINS ON 32 DATASETS. 187 ARE CACHED. +0.0073.

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
