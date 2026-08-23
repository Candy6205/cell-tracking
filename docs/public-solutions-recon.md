# Public-solution recon — 2026-08-20

> **2026-08-20, later the same day: the connector now authenticates.** The
> leaderboard endpoint that was blocked below is readable, and it corrects this
> file's central estimate — the live frontier is **0.957 / 0.935 at #20**, not
> 0.914-0.929. Everything sourced from the forum still stands; only the
> badge-derived frontier was wrong. See `strategy.md`.
>
> Produced from the Kaggle MCP connector, which at the time read **unauthenticated**.
> Supersedes several assumptions in `progress.md` and closes one open question
> in `holdout-split.md`. Items that contradict the progress log are marked
> **[SUPERSEDES]**.

---

## What the connector can and cannot do

| Works unauthenticated | Blocked (`Unauthenticated`) |
|---|---|
| `get_competition` | `get_competition_leaderboard` |
| `get_forum_topic` — full thread + all comments as raw markdown | `search_notebooks` |
| `list_competition_topics` | every `download_*` |
| `search_content` — returns each notebook's `best_public_score` | |
| `get_notebook_info` — **complete notebook source by slug** | |

Reading a public notebook's full source by slug is the significant capability.
No JS rendering, no scraping.

---

## 1. The division metric was exploited, patched, and rescored

This is the single most important finding and it reframes the public
leaderboard.

| Date | Event | Source |
|---|---|---|
| 2026-07-16 | Host Goldsborough: "the division metric can be gamed. We're working on a patch" | /723655 |
| 2026-07-17 | Patch lands, **commit `aa65e90`** | /733877 |
| 2026-07-17 | Exploit reported publicly: `outwrest/metric-hack-minimal-baseline-tta-2gpu` | /714101 |
| 2026-07-18 | Host: "an exploit of our metrics was found… We have made a patch and will re-score all submissions" | /727154 |
| 2026-07-22 | Kaggle staff: "All submissions are currently being rescored… most submissions will have at least a minor drop" | /728324 |
| 2026-07-23 | Rescore complete | /728324 |

**The exploit.** A synthetic "hub" node at sentinel coordinates
(`t=-1000`, `z=y=x=-10000`) welded to every real track root, plus a chain of
~20 fake divisions. The pre-patch division metric matched on **weak
connectivity** and ignored unmatched forks, so the hub satisfied the topology
conditions for free and cost no false positives.

**What `aa65e90` closed** — participant-reported (Luka Duvanov, /733877), never
documented by the host:

1. divisions require **directed local topology**, not shared weak connectivity
2. edges spanning more than one frame are **dropped**
3. merged edges are **collapsed**

**[SUPERSEDES] Those are the same three rules we reverse-engineered from
`tracking_cellmot/metrics.py` on 2026-08-18.** Our local scorer is therefore
aligned with the patched metric. The 2026-08-18 scorer work was independent
rediscovery of a public patch — corroboration, not new ground.

Post-patch the trick is dead: hub + 40 fake forks moved a one-to-one linker
0.9839 -> 0.9794, i.e. **-0.005**.

### The stale-badge trap

Public **notebook badge scores were apparently never corrected** after the
rescore.

**Any public notebook badge dated on or before 2026-07-23 may be a pre-patch
number.**

| notebook | badge | last updated | verdict |
|---|---|---|---|
| `kaiwalyaatulraut/biohub-cell-tracking-solution` | 0.966 | **2026-07-22** | **distrust — pre-rescore** |
| `boristown/dark-agi-...-solution` (fork of above) | 0.952 | 2026-07-19 | distrust — pre-rescore |
| `kunaldesale2408/biohub-cell-tracking` | 0.917 | 2026-08-18 | trustworthy — **full source read 2026-08-22, see addendum at end** |
| `yusuketogashi/no-hack-...-3rd` | 0.915 | 2026-08-05 | trustworthy |

Both "leaders" carry the hub. Estimated exploit contribution to the 0.966:
**+0.085-0.095**, putting its real edge Jaccard near **0.87-0.88** — roughly
where we already are.

~~**[SUPERSEDES] The live public frontier is ~0.914-0.929, not 0.966.** The real
gap from our 0.876 is **+0.04-0.05**, not +0.09.~~

**Corrected 2026-08-20 from the leaderboard itself: the frontier is 0.957 at #1
and 0.935 at #20.** Both statements above were inferred from notebook badges
because `get_competition_leaderboard` was blocked; the inference was low by
0.02-0.03. The real gap from 0.876 is **+0.059 to the top 20**. The claim that
the 0.966 badge is pre-patch is unaffected — it is simply not the frontier.

---

## 2. The 0.051 offset probably is not contamination

**[SUPERSEDES] `holdout-split.md`'s open question — and the offset's leading
hypothesis.**

**a. Our holdout is the structurally easiest slice that exists.** A participant
byte-compared every consecutive frame pair across all 199 training movies
(/724283, measured): `6bba` has a **7.47% duplicate-frame rate, affecting 89.1%
of its movies. `44b6` has exactly zero.** A frozen frame is a free link. Our
holdout is 12 `6bba_` datasets.

**b. The hidden test is a third embryo.** Host Thibaut Goldsborough confirmed
**"no overlap in embryo_ids between train and test"** (/716793). The test set
contains neither `44b6` nor `6bba`.

So the offset is at least partly a *distribution* gap, not a leakage gap — and
a `44b6`-only holdout is the cheapest available diagnostic.

Related trap: on a frozen pair in `6bba_fc516dc6` the volumes are byte-identical
but the GT edge still moves 8.90 um — past the 7 um match radius. Copying
detections across frozen frames does **not** work (/729082).

---

## 3. The runtime question is answered

**[SUPERSEDES] "Settle the hidden test size / runtime budget" — closed.**

- Hidden test is **~199-200 volumes** (official overview text quoted at /717228)
- Measured visible->submission runtime ratios cluster at **40-50x**; one report:
  637 s visible -> 7 h submission
- **9-12 h is normal** for learned pipelines; real timeouts observed
- Scoring itself is ~1 minute — the whole budget is inference

**Do not frame-stride to fit.** GT edges only ever span `t -> t+1`, so strided
edges are structurally unmatchable and score exactly 0.0 with no error raised
(/728613).

Our 0.876 run fits. Every stage added from here spends against a real ceiling.

---

## 4. Divisions: our own recorded direction, now with a measured number

Our recorded direction is **appearance-based division recovery** — find cells
that appear from nowhere — written into the `progress.md` to-do list before this
recon ran. That is the same idea as the orphan-node rule below. What follows is
confirmation of that direction, plus the two things we did not have: a selection
rule and a measured value.

### The rule, as the public tier implements it

Keep the linker strictly 1:1, then add forks in a **separate pass afterwards**:

- parent must already have exactly one child
- second daughter chosen only among **orphan nodes** (in-degree 0) at `t+1`
- greedy by `d(parent, orphan) + 0.15 * d(child1, orphan)`
- capped by a per-movie budget (caps seen: `0.0076`, `0.00375` of node count)

Because it only adds edges *into parentless nodes*, it cannot displace a true
positive. That is the structural property that makes it safe, and it is the
part worth copying.

### The measurement

Kunal Desale publishes his own numbers from the official scorer:
division TP/FP/FN `0/0/14` -> `6/31/8`, worth **+0.0129**, edge term unharmed.

Treat **+0.0129** as the credible target.

**[SUPERSEDES] The `pool_kernel_um = 5.0` NMS hypothesis is dead.** Measured
sister separation on real lineage edges is **7.24 um** (/732103) — daughters are
resolvable past a 5 um kernel. Consistent with our own measurement
(`division_features.py`: p50 9.41 um at t+1, only 6.1% below 5 um).

---

## 5. The 0.915-0.917 plateau: one recipe, two detectors

Kunal's detector is genuinely his own (a small `UNet3D(base=24)`, no official
repo, no ILP, no tracksdata). But his post-link graph patches are a
**line-for-line port of pilkwang's `BIOHUB_*` post-processor** — the same code
Togashi runs. The non-round constants match digit for digit: division caps
`0.0076` / `0.00375`, sister weight `0.15`, linefit `0.8` / window `2`,
short-component minimum `6`, link gates `6.0` / `10.0` um, velocity weight
`0.5`, snap radius `3.2` um.

Three things worth knowing:

- **Togashi discards the ILP's edge assignment.** He runs our exact flags, then
  `edges = motion_edges` — a fresh two-gate Hungarian pass. The ILP survives
  only as a node set and a <=1.0 um probability bonus.
- **[SUPERSEDES] DeepCenter does not matter.** Togashi carries ~15 DeepCenter
  env vars and disables every one; the dataset is not even attached. That also
  retires the DeepCenter-contamination question entirely.
- **Neither reads `estimated_number_of_nodes`.** Togashi's only calibration is
  four hard-coded counts taken from the visible 4-dataset test — which will not
  survive the swap. `cell_count.py` is not redundant.

---

## 6. Legitimate transplants, best first

From the 0.966-vs-0.952 diff. The exploit is **common-mode across both**, so it
cancels in the subtraction and these structural items stand. Realistic combined
ceiling **+0.015-0.03**, not +0.09.

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | **Candidate edges: per-target top-3 parents at >=0.25, plus a 10 um gate** (vs bare `> 0.5`) | Under a softmax over sources, any target whose best parent scores 0.25-0.50 is stranded. This is the "detected but unlinked" bucket. | small |
| 2 | **`MIN_TRACK_LEN = 3` component pruning**, with division and boundary exemptions | Attacks the node-count penalty directly | ~30 lines pandas, no GPU |
| 3 | **`det-threshold` 0.99 -> 0.95** | Already on the to-do list as a sweep | one flag |
| 4 | **TTA-average the U-Net features fed to the linker** (`.mean(dim=0)`) | The fork uses view 0 only and loses ground | moderate, costs runtime |
| 5 | **One-frame gap closing** — Hungarian <=10 um, intensity-refined synthetic midpoints | Patch-compatible by construction: inserts a middle node rather than a `t -> t+2` edge | moderate |
| 6 | **Orphan-node forks** (section 4) | +0.0129 measured by its author | ~40 lines |

Items 2 and 6 need **no GPU run** — both are pandas over predictions we already
have. Measure them first.

Negative result worth having: the 0.952's author swept scalars extensively and
posted the log — 0.950 / 0.950 / 0.946 / 0.949 / 0.950. **Global knob-tuning is
exhausted.** The remaining gains are structural.

---

## 7. Free diagnostic

`summarise()` drops the division term entirely when a submission contains no
forks. Our submission has none — therefore **our 0.876 IS our pure adjusted edge

**CORRECTED 2026-08-20.** The number is right, the reason is wrong.
`summarise()` drops the division term only when the WHOLE split has no
divisions at all — `metrics.py`:

    division_total = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    if division_total == 0:
        division_jaccard = float("nan")
        score = adj_edge_jaccard

The hidden test has ground-truth divisions, so `division_fn > 0` and the term
is **included at zero**, not dropped. Same number; different meaning. The
maximum score is **1.1**, not 1.0, and we forfeit the whole 0.1.

Jaccard on the hidden test**, with no ablation needed.

---

## Revised to-do

| Priority | Action | Cost |
|---|---|---|
| 1 | **Orphan forks** + **`MIN_TRACK_LEN` pruning** on the 12 `.geff` we already have | CPU only, no rerun |
| 2 | Score a **`44b6`-only holdout** and compare to the `6bba` 0.9269 | one inference pass |
| 3 | Transplant **top-3 candidate parents at >=0.25** | small code change + rerun |
| 4 | `det-threshold` sweep, extended down to 0.95 | one flag |
| ~~—~~ | ~~Re-run the 0.916 notebook on our 12 holdout datasets~~ | **drop** — those badges are pre-patch |
| ~~—~~ | ~~`pool_kernel_um` sister-separation check~~ | **closed** — 7.24 um |
| ~~—~~ | ~~Settle hidden test size / runtime~~ | **closed** — ~199-200 volumes, 9-12 h |

---

*Written 2026-08-20. Sources are Kaggle discussion IDs under
`https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/`
and public notebook source read via the Kaggle connector.*

---

## Addendum 2026-08-22 — kunaldesale2408 (0.917) full-source anatomy

The first complete recipe read from the trustworthy tier — and it does NOT
use the official pipeline at all:

- **Self-trained detector**: classic 3D UNet (base 24), weights from
  `xiaoleilian/biohub-unet3d-weights` with `val_recall` stored in the
  checkpoint. Input: 4x-pooled Y/X, percentile-normalised (p50..p99.5),
  optional top-hat preprocessing per model. 2-3 model ensemble, heatmaps
  AVERAGED (their comment: unioning detections measured 0.661 — the
  node-count penalty punishes it). Flip-quartet TTA on logits. Peak
  threshold 0.15, centroid refinement at full resolution, physical NMS
  4.0 um. **Confirms the 0.9+ tier trains its own models.**
- **Linker**: Hungarian, two-stage gate (6 um tight then 10 um), cost =
  distance to a VELOCITY-EXTRAPOLATED position (0.5 x previous velocity)
  plus an APPEARANCE term (2.0 x |logit(det score) diff|).
- **Post patches, VAL-24 with the official metric**: base 0.8387 ->
  +safe divisions 0.8516 (+0.013) -> +1-frame gap closing (snap-only)
  0.8518. Their division patch stacks filters ours lacks: orphan must be
  the sister's MUTUAL nearest orphan, both children must continue at t+2,
  and sisters must DIVERGE (d(t+2) - d(t+1) >= 2.25 um); per-frame and
  global caps. Gap closing never synthesises a midpoint — it only snaps
  to an unused low-score (0.10..0.15) detection candidate within 3.2 um.
- **5-frame linefit smoothing** (LINEFIT_WINDOW=2, weight 0.8) of node
  coordinates along single-child chains — pure post-processing.

Transplant candidates for OUR pipeline, cheapest first: (1) add
mutual-nearest + divergence filters to the orphan-fork patch (could cut
the 13 visible div FPs); (2) velocity extrapolation in the ILP/linker
cost; (3) appearance term from detection confidence; (4) linefit
smoothing (helps the 7 um match radius on fast movers).

## Addendum 2026-08-22b — yusuketogashi (0.915) post-processing anatomy

Read from the full source (function skeleton + key segments; 225 KB
notebook). Its stack, in pipeline order:

1. **Learned edge re-ranker** (`motion_relink_edges` + a preloaded
   LOCAL_ASSOCIATION_RANKER artifact): candidate edges scored by ~25
   engineered features — primary edge_prob, raw vs motion-extrapolated
   distance and their difference ("motion gain"), per-axis deltas,
   **7 um local density at both endpoints**, in/out degrees, frame cell
   counts, candidate rank/count, velocity magnitude, t_norm — fed to an
   offline-trained ranker, then gated Hungarian assignment passes.
2. **Two-tier gap closing**: 1-frame and strict 2-frame, with SYNTHETIC
   midpoints refined by local intensity (`refine_synthetic_midpoint`) —
   the opposite of kunaldesale's snap-only, but insured by:
3. **DeepCenter as a veto detector**: synthetic/repair points must pass a
   second independent detector's heatmap
   (`deepcenter_accept_repair_point`) before entering the graph.
4. **Acceleration lookahead** in motion prediction (bonus for consistent
   acceleration), not just velocity.
5. Safe divisions post-link, short-track component filter, linefit
   smoothing (same family as kunaldesale's).
6. **Edge TTA, 4 views** (BIOHUB_EDGE_TTA_VIEWS=4) — TTA on the edge
   head, not only detection.

For US: vel-sweep and orphan-v2 (in flight) are the entry-level versions
of items 1 and 5. The learned re-ranker is the standout transplant: our
cache already holds probs/coords, the other features are computable, GT
edges outside the holdout supply labels, and a gradient-boosted ranker
trains on CPU. Synthetic-midpoint gap closing becomes viable when paired
with a second-detector veto (we have DeepCenter in the pack, unused).
