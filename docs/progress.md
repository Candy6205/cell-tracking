# Progress log — Biohub Cell Tracking During Development

> What is done, what is confirmed, what is next, and what already bit us.
> Engineering conventions live in `docs/engineering-rules.md`
> (rule 1: **degradation is never silent**).

---

## Status (2026-08-19) — LB 0.876, local 0.9269

**Submitted and scored. Public LB 0.876**, up from 0.681 (the classical DoG
pipeline). The configuration is the official repo plus pilkwang's public weights
and two command-line flags — no DeepCenter, no second seed, none of the ~40
`BIOHUB_*` knobs.

| | LB | local, 12 held-out `6bba` |
|---|---|---|
| classical DoG pipeline | 0.681 | ~0.45 |
| **official + ILP + app 0.0 / disapp 1.5** | **0.876** | 0.9269 |
| the public 0.916 notebook (full stack) | 0.916 | not measured with the fixed scorer |

### The offset: local is optimistic by 0.051

This is the number every future decision rests on. Candidate causes, none yet
separated:

1. **Contamination.** The 12 `6bba` datasets are provably unseen by *DeepCenter*
   (its `split_manifest.json` says so) but the **primary** U-Net's split is still
   unknown — see `docs/holdout-split.md`. If it trained on them, 0.9269 is
   inflated.
2. **The hidden test is harder or differently distributed.** Density drives the
   score hard (`corr(edge J, true density) = -0.68`), so a denser test set alone
   would explain a large gap.
3. **Node-count penalty.** Locally it is 0.0009 because ILP trims nodes to just
   under `N_true`. On a test set with a different density mix the same trimming
   could overshoot or undershoot.

**Operationally:** use local scores for **paired deltas**, never for absolute
targets. A change worth +0.01 locally is very likely worth something on the LB;
"local 0.93 so we are near 0.93" is false.

To pass the public 0.916 notebook we need roughly **+0.04 more on the LB**. That
is also, to within noise, what its whole post-processing stack is worth over our
bare configuration — so the shortest path is to find which of its ~40 knobs
actually carry that 0.04, rather than adopting all of them.

## The local best: 0.9269

**12 held-out `6bba_` datasets, official `scripts/evaluate.py`, official repo
plus pilkwang's public weights. No DeepCenter, no second seed, none of the
`BIOHUB_*` post-processing.**

```
python scripts/predict_unet_transformer.py \
    --splits splits.json --split 0 \
    --weights .../unet_transformer/split_0/edge_predictor_best.pth \
    --det-threshold 0.99 --use-ilp \
    --ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5 \
    --evaluate
```

```
score              0.9269
adj_edge_jaccard   0.9269
edge_jaccard       0.9278     (TP/FP/FN 9863 / 318 / 450)
division_jaccard   0.0000     (TP/FP/FN    0 /   0 /  12)
node_recall        0.9882
node-count penalty 0.0009     (was 0.0140)
```

**+0.0422 over the greedy default, and it improved 12 of 12 datasets.**

| step | score | delta |
|---|---|---|
| greedy (repo default) | 0.8847 | — |
| `--use-ilp` | ~0.907 (3-dataset estimate) | +0.02 |
| **+ appearance 0.0, disappearance 1.5** | **0.9269** | **+0.0422** |

Where the gain came from:

```
edge_jaccard        0.8960 -> 0.9278   (+0.0318)
node-count penalty  0.0140 -> 0.0009   (+0.0131 to the score)
division term       0.0027 -> 0.0000   (-0.0027)
```

TP +80, **FP -288 (-48%)**, FN -80, `n_pred` 191,512 -> 168,675 (-12%).

| dataset | greedy J | ILP+A J | delta |
|---|---|---|---|
| 6bba_268e1230 | 0.9016 | 0.9529 | +0.0514 |
| 6bba_5c824876 | 0.9033 | 0.9447 | +0.0414 |
| 6bba_7af54fde | 0.8280 | 0.8692 | +0.0411 |
| 6bba_c27cba08 | 0.9107 | 0.9482 | +0.0375 |
| 6bba_3abfe10a | 0.7705 | 0.8055 | +0.0349 |
| 6bba_283bf9f1 | 0.9171 | 0.9519 | +0.0348 |
| 6bba_3a1849c2 | 0.8783 | 0.9104 | +0.0321 |
| 6bba_afb141ff | 0.9070 | 0.9330 | +0.0260 |
| 6bba_61dd1e0d | 0.9164 | 0.9394 | +0.0230 |
| 6bba_7b5d3b2c | 0.9596 | 0.9803 | +0.0207 |
| 6bba_2312ac41 | 0.9611 | 0.9798 | +0.0187 |
| 6bba_aeee7805 | 0.9680 | 0.9768 | +0.0087 |

For reference the public LB of the 0.916 notebook — which stacks DeepCenter, a
second seed and ~40 `BIOHUB_*` post-processing knobs — is **0.916**. Different
data, so not directly comparable, but this configuration is a bare pipeline plus
two command-line flags.

### The previous baseline, for the record

Greedy linking, same weights, same `--det-threshold 0.99`:

```
score              0.8847
adj_edge_jaccard   0.8820
edge_jaccard       0.8960     (TP/FP/FN 9783 / 606 / 530)
division_jaccard   0.0265     (TP/FP/FN    6 / 214 /   6)
node_recall        0.9974
```

### What this says

1. **Detection is solved.** `node_recall = 0.9974`. Every point of edge loss is
   a *linking* error, not a missed cell. Our classical DoG detector reaches
   0.70 recall by comparison — the classical path is dead as a main line.
2. **Divisions.** Official `evaluate.py` reported division TP/FP/FN =
   **6 / 214 / 6**, and `division_jaccard = 0.0265` out of a possible 1.0.

   **CORRECTED 2026-08-20.** This entry used to read "220 forks predicted,
   12 real, 6 correct, 214 spurious -- 18x over-prediction". **That 220 is
   6 + 214.** It was never counted from the graph, and no independent fork
   count exists anywhere in this log, so "18x over-prediction" was circular.

   It matters because the official metric only counts a predicted fork at
   all when it is matched. Read from `division_metrics.py`:

   ```python
   fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks

   evaluable_forks = {
       pred_id for pred_id in pred_forks
       if pred_id in pred_to_gt and gt_graph.out_degree(pred_to_gt[pred_id]) >= 1
   }
   ```

   `evaluable_forks` needs a matched parent with GT out-degree >= 1;
   `considered` only holds forks inside a GT division's local window; and
   `invalid_forks` (cross-component, malformed) need matched descendants as
   evidence. **A fork lying entirely among unannotated cells enters none of
   the three and is not penalised** -- the same leniency the node and edge
   terms already have.

   So the greedy graph may well have held thousands of forks, of which 220
   were merely the ones the metric could see. The true over-prediction
   factor is unknown.

   **Do not read this as an invitation to spam forks over unannotated
   regions.** It gains nothing -- `div_J = TP/(TP+FP+FN)` and unmatched
   forks add no TP -- and it is the same genre as the hub-node exploit that
   was patched on 2026-07-17 and rescored. What the leniency legitimately
   buys is room to run a division rule with decent recall and ordinary
   precision, instead of needing surgical precision.
3. The node-count penalty costs `0.8960 - 0.8820 = 0.014`.

### Division head-room, computed

**These are the greedy run's numbers** (division TP/FP/FN = 6/214/6). With
`FN = 6` already lost, even perfect precision caps `division_jaccard` at 0.5
*for that configuration*.

The submitted ILP+A configuration is a different starting point: it emits **no
forks at all**, so its division line is `0/0/12` — ceiling 1.0, worth +0.1 of
score, but every one of the 12 has to be recovered from scratch. The realistic
target is the orphan-fork pass's measured **+0.0129**, not either ceiling.

| division FP | division_jaccard | score gain |
|---|---|---|
| 214 (now) | 0.0265 | — |
| 50 | 0.0968 | +0.007 |
| 20 | 0.1875 | **+0.016** |
| 6 | 0.3333 | **+0.031** |
| 0 | 0.5000 | **+0.047** |

Going past +0.047 also requires recovering the 6 missed divisions.

### ✅ Our local scorer is now VERIFIED against the official one

Both run on the **same** 12 predicted graphs:

| | official `evaluate.py` | our `run_verify` |
|---|---|---|
| edge TP / FP / FN | 9783 / 606 / 530 | **9783 / 606 / 530** |
| edge_jaccard | 0.8960 | 0.89596 |
| adj_edge_jaccard | 0.8820 | 0.88201 |
| division TP/FP/FN | 6 / 214 / 6 | 6 / **231** / 6 |
| score | 0.8847 | 0.8845 |

**The edge term matches to the digit.** That is the licence to keep using the
fast local scorer for paired sweeps. The 0.0002 score difference is entirely the
division term, where our `score_divisions_simple` finds 231 spurious forks to
the official 214 — close, but the official number is the one to quote.

`edge_fp_lenient = 5`: of the 606 false positives, the old scorer would have
seen **five**.

### Per-dataset, sorted by true density

`N/N_true` comes from the geff's `estimated_number_of_nodes`, so true density =
`n_pred / 100 / (N/N_true)`.

| dataset | edge J | true cells/frame | N/N_true | div FP |
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

```
corr(edge J, true density) = -0.68
corr(edge J, N/N_true)     = -0.64
```

**`3abfe10a` is genuinely dense, not broken** — 62840 nodes at `N/N_true = 1.21`
means ~519 real cells per frame. Density is the primary failure axis.

`7af54fde` is the exception that names a second one: only 97 cells/frame yet
J = 0.828, and the worst over-detection at 1.28. Something other than density
hurts it — most likely motion (the `6bba` batch has a median displacement p99 of
7.81 um, already above the 7 um match radius).

So there are **two** failure modes to attack separately: crowding, and fast
movement.

### ⭐ ILP linking: +0.0295 over greedy (2026-08-18, evening)

Paired comparison, same weights, same `--det-threshold 0.99`, on three held-out
`6bba` datasets (`3abfe10a` 519 cells/frame, `2312ac41` 119, `aeee7805` 50):

| config | edge J | adj_J | node penalty | div_J | **score** |
|---|---|---|---|---|---|
| greedy (repo default) | 0.8956 | 0.8826 | 0.0130 | 0.0308 | 0.8857 |
| `--use-ilp` | 0.9150 | 0.9073 | 0.0077 | 0.0000 | 0.9073 |
| **+ app 0.0, disapp 1.5** | **0.9183** | **0.9152** | **0.0031** | 0.0000 | **0.9152** |
| + division-weight 0.3 | 0.8999 | 0.8907 | 0.0092 | 0.0308 | 0.8937 |
| + disapp 3.0 | 0.9097 | 0.9120 | **-0.0023** | 0.0000 | 0.9120 |

Per dataset, greedy -> ILP+A:

| | TP/FP/FN greedy | TP/FP/FN ILP+A | n_pred |
|---|---|---|---|
| 2312ac41 | 1113/26/19 | 1116/7/16 | 13594 -> 12460 |
| 3abfe10a | 816/124/119 | 825/126/110 | 62840 -> 54700 |
| aeee7805 | 757/14/11 | 757/7/11 | 5493 -> 5052 |

**Most of the gain is the node-count penalty, not the edge Jaccard.** edge J rose
only 0.0033 from ILP-default to ILP+A, while the penalty fell 0.0077 -> 0.0031.
Raising the disappearance cost makes the solver prefer long continuous tracks
and discard scattered nodes, so `n_pred` drops sharply. `node_recall` pays a
little (0.9952 -> 0.9900); the penalty saving more than covers it.

At disappearance 3.0 the penalty goes **negative** — `n_pred` falls below
`n_true` and the formula, which has no upper clamp, pays a bonus (`adj_J 0.9120`
> `edge_J 0.9097`). It still loses overall because `node_recall` drops to 0.9758.
So the optimum is near 1.5 and the curve is flat there (1.5 vs 3.0 differ by
0.003) — not worth refining.

### Divisions: with the CURRENT method, the negative feedback outweighs the positive

Scope of this result: **one knob, moved one notch, on three datasets containing
four ground-truth divisions.** That is enough to set the current default and
nothing more. It is not a verdict on divisions as a direction.

`--ilp-division-weight 0.3` produced division TP/FP/FN of exactly 2/61/2, digit
for digit the same as greedy. Lowering the cost does not buy *selective* forks
at this setting; it opens the floodgates. The trade:

```
division_jaccard   0 -> 0.0308        worth +0.0031 of score
adj_edge_jaccard   0.9152 -> 0.8907   costs  -0.0245
```

**8:1 against.** So `division_weight = 1.0` (zero forks) is the right default
for now: forfeit the division term, protect the edge term.

#### Divisions also corrupt the edge term — measured

A missed division is not only a lost 0.1-weighted point. The parent links to one
daughter and the other daughter's incoming edge is lost, so edges break too:

```
GT edges leaving a dividing cell : 24, failure rate 29.2%   (greedy run)
GT edges leaving a normal cell   : 10289, failure rate 5.1%
```

ILP currently emits **no forks at all**, so each parent keeps at most one
daughter and **at least half of those 24 edges must fail** — about 12 of the 530
total failures, i.e. 2.3% of them, worth roughly **+0.001** of adj_J if
recovered. Real, but small, because divisions are rare (1 per 854 edges).

#### Two suspected obstacles, one measured and one NOT

1. **Sample size — measured.** 151 divisions across all 199 datasets, 1 per 854
   edges, 56% of datasets have none. The training split ever sees ~137 division
   events.
2. **The NMS kernel may hide them — UNTESTED HYPOTHESIS.**
   `pool_kernel_um = 5.0`, so two peaks closer than 5 um collapse into one
   detection. Right after a split the daughters should be 1-2 um apart, and at
   the median drift of 1.82 um/frame would need ~3 frames to separate past
   5 um — outside the metric's ±1 frame tolerance. **This has not been checked
   against the data.** `division_gif_cell.py` prints the real sister-separation
   curve from GT and settles it in a minute of CPU.

#### What has NOT been tried

- any division weight other than 0.3 and 1.0, or any sweep at a sample size
  where `division_jaccard` means anything (needs ~50+ datasets)
- a smaller `pool_kernel_um`, globally or only near fork candidates
- deliberately reporting the split one or two frames late, to exploit the
  metric's ±1 frame window once the daughters are resolvable
- the official `division_metrics.py` — every division number we have locally
  comes from `score_divisions_simple`, a lenient approximation

An earlier note here called divisions "the biggest single lever". The headroom
is real; the price at the one setting tested is not worth paying.

### The decomposition

```
detect_ceiling  0.9953     GT edges with BOTH endpoints detected
link_rate       0.9530     of those, the fraction actually linked
```

Detection contributes 0.5% of the loss. **Linking is 95% of it.** Every hour
spent on detection is wasted; the linker and the division logic are the targets.

### ⚠️ Every local number from before 2026-08-18 is void

`score_one` had the wrong false-positive rule (details below). In particular the
previously recorded **"public 0.916 pipeline scores adj_J 0.9614 on 8 held-out
datasets" is dead** — it came from the broken scorer and cannot be compared to
the 0.8820 above. To learn what the post-processing pile is actually worth, the
0.916 notebook must be re-run on **these same 12 datasets** and scored with the
official `evaluate.py`.


---

## ⭐⭐ The edge threshold is the binding constraint (2026-08-20)

Read verbatim from `scripts/predict_unet_transformer.py` in the official repo.
This is the largest identified lever and it has never been touched.

### Three facts, from the source

**1. `threshold` defaults to 0.5 and our submission never sets it.**

```python
    det_threshold: float = 0.5
    det_tta: bool = True                 # already ON by default
    pool_kernel_um: float = 3.0          # NOT 5.0 as recorded earlier
    edge_activation: str = "softmax"
    threshold: float = 0.5               # <- the edge candidate gate
```

**2. With ILP on, the greedy capacity caps are disabled entirely.**

```python
    def __post_init__(self) -> None:
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2
```

Both stay `None` when `use_ilp=True`, and the greedy loop guards every cap with
`is not None`. So the ILP does receive an unconstrained candidate set — but only
of pairs that already cleared `threshold`:

```python
    candidates = sorted(
        [(probs[i, j], i, j)
         for i in range(n_src) for j in range(n_tgt)
         if probs[i, j] > cfg.threshold],
        reverse=True,
    )
```

**`threshold` is therefore the ONLY thing limiting what the linker may consider.**

**3. The softmax normalises over sources**, i.e. down the column: for each target
`j`, `sum_i probs[i, j] = 1`.

### Why that combination loses edges structurally

A target keeps a candidate only if its best parent holds **more than half** of
the column mass. One unambiguous parent -> p ~ 1.0, fine. Two plausible parents
-> ~0.5/0.5, and **both fall below the gate, so the target gets no candidate
edge at all**. Three or more -> guaranteed loss.

This predicts, without any new measurement, everything already in this log:

| observation | explained |
|---|---|
| 60% of edge failures are `DROPPED` — both endpoints detected, no edge emitted | targets whose mass split below 0.5 |
| `corr(edge J, true density) = -0.68` | more neighbours -> more competing parents -> more mass splitting |
| `3abfe10a`, 519 cells/frame, worst J at 0.7705 | the densest movie is the most split |
| `detect_ceiling 0.9953` but `link_rate 0.9530` | detection is fine; the gate throws away links |

### The headroom, arithmetically

Current 12-dataset holdout: TP/FP/FN = 9863 / 318 / 450, J = 0.9278.

A missed link costs twice (an FN, and the predicted edge leaving its matched
predecessor becomes an FP), so recovering a dropped edge moves both terms. If
the `DROPPED` bucket is ~60% of the 450 FN and each recovery also retires one
FP:

```
TP 10133 / FP ~150 / FN ~180  ->  J = 0.9685   (+0.041)
```

That is an upper bound, not a forecast. But it is the right order of magnitude,
it is one config value, and the whole LB gap is +0.059.

### The coupling that must be swept with it

`ilp_appearance_weight = 0.0` makes starting a track free while
`ilp_disappearance_weight = 1.5` makes ending one expensive. Lowering
`threshold` hands the solver cheap ways to avoid that 1.5, so it will link more
aggressively and may buy FPs. **Sweep `threshold` x `disappearance_weight`
together** — rule 15, parameters interact.

Forks stay impossible throughout: a fork needs
`edge_prob > division_weight - appearance_weight = 1.0`. Lowering the gate
cannot create spurious divisions, so this is safe with respect to the division
term. Divisions remain a separate post-hoc pass.

### Why this changes how we spend GPU

`sweep_edge_threshold.py`'s monotonicity argument holds for **greedy only** —
with ILP, a larger candidate set is a genuinely different optimisation, not a
prefix of the old one. So the sweep cannot be replayed from an existing `.geff`.

The fix is to cache one level earlier. `predict_video` is the only GPU stage;
`build_graph` + `ILPSolver.solve` are pure CPU. Run inference **once** at
`threshold=0.05`, dump `(coords, edges, probs)` per dataset, and every linking
experiment afterwards — threshold, appearance/disappearance, orphan forks,
component pruning, gap closing — is a CPU replay costing no quota at all.

**Build the cache before running any sweep.**
---

## The candidate cache, and what it measured (2026-08-20)

The GPU stage now runs once and every linking experiment replays on CPU.
`cache_candidates.py` dumps the raw candidates, `link_replay.py` re-solves
them, `run_sweep.py` is the Kaggle entry point. Toolchain, and the five traps
that cost a run each, in `docs/kaggle-toolchain.md`.

### The cache is faithful — verified, not assumed

Replaying the submitted configuration (threshold 0.50, appearance 0.0,
disappearance 1.5) out of the cache:

```
edge TP/FP/FN    9863 / 318 / 450        <- recorded 9863 / 318 / 450
adj_edge_jaccard 0.9269                  <- recorded 0.9269
```

Digit for digit. **So a sweep over the cache measures the real pipeline**, and
one full 12-dataset pass costs **154 s of CPU** against 11 min of GPU.

### The correction it produced: 0.9953 / 0.9530 is the GREEDY decomposition

The recorded `detect_ceiling 0.9953 / link_rate 0.9530` sits in this log where
it reads as describing the current configuration. It does not:

| | detect_ceiling | link_rate |
|---|---|---|
| as recorded | 0.9953 | 0.9530 |
| ILP+A, measured 2026-08-20 | **0.9823** | **0.9736** |

The recorded pair reproduces exactly from the greedy run's TP of 9783
(9783 / 0.9530 = 10265 reachable, and 10265 / 10313 GT edges = 0.9953), so it
is greedy's, not ILP+A's.

**This is how ILP actually wins**: it gives up a little detection reach
(0.9953 -> 0.9823, because the solver discards nodes) to buy a lot of linking
(0.9530 -> 0.9736). That trade was not visible before.

Detection's share of the loss is therefore **1.8%, not 0.5%** — still small,
still linking-dominated, but 3.5x what this log claimed.

### The measurement: 11.8% of targets have no candidate parent at all

Pooled over the 12 held-out `6bba` datasets, dumping at threshold 0.05:

| | value |
|---|---|
| candidates above 0.5 — everything the 0.876 submission ever saw | 167,398 |
| candidates above 0.05 | 338,781 |
| **ratio** | **2.02x** |
| targets | 189,872 |
| **targets with NO candidate above 0.5** | **22,474 = 11.8%** |

The softmax normalises over sources, so at most one parent per target can
exceed 0.5. `above_0.5` is therefore exactly the count of targets that had any
candidate at all, and the identity holds in the data (`above_0.5 <= targets`
in all 12). Those 22,474 targets are **structural false negatives**: no
linker, however good, could have connected them.

### It is driven by over-detection, NOT density

Per-dataset stranded fraction against the statistics already in this log:

| stranded fraction vs | Pearson r |
|---|---|
| **`N/N_true` (over-detection)** | **+0.79** |
| true cells/frame (density) | **-0.07** |
| edge Jaccard | -0.53 |

**Density is uncorrelated.** This log's "density is the primary failure axis"
holds for the score, but not for the mechanism behind the largest error
bucket. The plausible mechanism: a spurious detection has no true parent, so
its softmax column spreads over whatever is nearby and nothing clears 0.5.

### Which cuts both ways — read this before spending on it

A stranded target that is a **spurious detection should stay unlinked**;
forcing an edge into it manufactures a false positive. The +0.79 correlation
says a substantial share of the 22,474 are exactly that. So the 2.02x is
**not** free headroom, and the "+0.041 upper bound" in the section above is
optimistic — it assumed every stranded target was a real cell.

Separating the two is precisely what `appearance_weight` and
`disappearance_weight` arbitrate, which is why the sweep is 2-D.

### The line this opens: over-detection has never been measured

`detect_ceiling = 0.9823` counts **missed** cells. It says nothing about
**spurious** ones. `N/N_true` runs 1.01 to 1.30 across the holdout, so up to
30% of predicted nodes are invented. They pay the node-count penalty, and they
now appear to drive the largest edge-failure bucket as well.

"Detection is solved, every hour spent on it is wasted" is therefore **half
right**. Recall is solved. Precision was never measured. Raising
`det-threshold` above 0.99, or tightening NMS, is live and untried.
### Counted, not estimated: what the gate actually costs

`stranded_audit.py`, run 2026-08-20 on the cache plus ground truth. Two
measurements, and they point in opposite directions, which is the point.

**1. The stranded targets are disproportionately spurious.**

| | matches a GT node within 7 um |
|---|---|
| targets with a confident parent (167,398) | **6.1%** |
| stranded targets (22,474) | **1.5%** |
| ratio | **0.24** |

Sparse annotation depresses both rates equally, so the ratio is the signal,
and it holds in every one of the 12 datasets (0.11 to 0.55). Most of the
22,474 are detections the model invented. They have no true parent and
**should** stay unlinked.

**2. And yet 278 real edges are stuck behind the gate.**

```
GT edges                       10313
  target cell never detected      27
  target detected but STRANDED   278
```

**278 of the recorded 450 false negatives — 62% — exist because of the 0.5
gate.** Not an upper bound; that is the count.

### The arithmetic that decides it

Recovering all 278 while manufacturing F new false positives gives
`J = 10141 / (10631 + F)` against the current 0.9278:

| F | J | delta |
|---|---|---|
| 0 | 0.9539 | **+0.026** |
| 278 | 0.9296 | +0.002 |
| 500 | 0.9107 | **-0.017** |

**Break-even is near F = 300**, and the pool the solver may draw from holds
22,196 spurious stranded targets. The 278 real edges are **1.2%** of it. So
lowering the gate is a needle-in-haystack problem, and whether it pays comes
down entirely to how selective the ILP is.

That sharpens the sweep's prediction: `appearance_weight = 0.0` makes
starting a track free while `disappearance_weight = 1.5` penalises ending
one, so the solver is currently biased *towards* linking. At a lower gate
that bias is a liability. **Expect the winning cell to pair a lower
threshold with a LOWER disappearance weight**, not the 1.5 we submitted.

### Why over-detection is now the higher-leverage target

Those 22,474 spurious nodes are charged twice: they pay the node-count
penalty, and they bury the 278 recoverable edges in noise.

So reducing over-detection is not merely worth its own small penalty saving
— **it is the precondition that makes the threshold lever safe**. Halve the
spurious pool and the needle-in-haystack ratio halves with it, moving
break-even in our favour. The two lines multiply; they do not add.

The project has never tried it, because `detect_ceiling` reads 0.98-0.99 and
that was taken to mean detection was solved. It counts **missed** cells only.
`N/N_true` of 1.01 to 1.30 says up to 30% of predicted nodes are invented,
and nothing in this log has ever measured that as a cost.

**Next on this line:** sweep `det-threshold` upward from 0.99 and watch the
stranded count, not just the score. It is a cached-replay quantity for the
threshold part, but det-threshold changes detection itself, so it needs one
inference pass per value — about 11 minutes each.

---

## det-threshold is not the lever, and the ceiling is the model (2026-08-21)

Scored on the cache at the converged linking optimum (threshold 0.40,
disappearance 3.5), so only `det_threshold` varies:

| det | adj_J | edge_J | detect_ceiling | link_rate | nodes |
|---|---|---|---|---|---|
| **0.99** | **0.9387** | 0.9363 | 0.9825 | 0.9839 | 161,649 |
| 0.98 | 0.9375 | 0.9362 | 0.9827 | 0.9836 | 164,185 |
| 0.95 | 0.9367 | 0.9365 | 0.9821 | 0.9842 | 166,209 |

**`detect_ceiling` does not move** -- 0.9825 / 0.9827 / 0.9821 -- and neither
does `edge_J`. The fall in `adj_J` is entirely the node-count penalty on 2.8%
more nodes.

So the extra detections admitted by a lower gate are **not** the missing
endpoints of the 180 unreachable GT edges. Those cells are not peaks in the
heatmap at any threshold down to 0.95. The detection ceiling is set by the
model's output, not by our threshold.

### Three wrong directions on one knob

For the record, because the pattern matters more than the knob:

1. **Raise it** — argued from the stranded pool being 98% spurious. Counted
   nodes, never scored.
2. **Lower it** — argued from detection being 52% of the loss and from a
   claim that sparse annotation deflates the sigmoid. The deflation claim was
   wrong: the negative term carries total weight 0.1 across ~4.19M voxels, so
   an unannotated cell is pulled toward zero with weight ~2.4e-8.
3. **Lower it** — argued from the metric alone: over-detection costs
   `0.1 x over-prediction` and nothing else, while a miss costs an FN plus an
   induced FP. This reasoning is sound and the direction still lost.

Reasoning (3) is the only one that did not depend on a story about the
model's internals, and it is the only one that would have been safe to act on
without measuring. It was still not enough. **Score the thing.**

### What it leaves open

The threshold/disappearance pair converged, and it only ever moved loss
between detection and linking. `det_threshold` cannot raise the ceiling. What
remains for the edge term is a **better detector**, which means training --
and `docs/strategy.md` records the host permitting Zebrahub, whose ZSNS001
carries 21,697,591 annotated nodes and 635,042 divisions against this
competition's 133,318 and 151.

---

## PARKED: forks the solver refused (2026-08-21)

> **Priority ruling, from the project owner, 2026-08-21: divisions come after
> the edge and node predictions are good enough. Do not pick this up as the
> next thing to run.** The live thread is Zebrahub — a better detector and
> edge head. This section is written down so it does not have to be
> rediscovered, not because it is next.
>
> **`division_weight = 1.0` is a deliberate decision, not an oversight.**
> Lowering it was measured 8:1 against (divJ 0 -> 0.0308 buys +0.0031 while
> adj_J loses 0.0245) and the ban was kept on purpose to protect the edge
> term. Everything below works *with* that ban and never asks to relax it.

Everything this log has written about divisions assumed the missing daughter
was missing because the model never scored her. `division_ceiling.py` measured
it, and the assumption is wrong.

### The premise, measured

| | |
|---|---|
| GT divisions in the 12-dataset holdout | 12 |
| **with BOTH daughters carrying a candidate above 0.5** | **9** |
| orphan nodes before the solve | 24,114 |
| orphan nodes after the solve | **11,016** |

The first row is the one that changes the plan. The edge head applies
`softmax(dim=0)`, normalising **down the target column**, so both daughters can
score above 0.5 at the same time without competing — they are different
columns. In 9 of 12 cases they do.

So the division information is in the candidate set already. What discards it
is one line of solver configuration:

```
a fork is taken iff   edge_prob > division_weight - appearance_weight
we submit             division_weight 1.0,  appearance_weight 0.0
therefore a fork needs edge_prob > 1.0      <- impossible
```

The parent keeps one daughter, the other daughter's confident incoming edge is
thrown away, and she ends the solve with in-degree 0.

**This is not the stranded pool.** `stranded_audit.py` found the pre-solve
stranded targets to be ~98.5% spurious, and that number has been the standing
argument against every orphan-fork idea. It does not apply to these: a declined
daughter has a *confident* edge, from a node the solver kept, and the ILP has
already discarded more than half the junk (24,114 -> 11,016) before any fork
rule sees it.

### `fork_pass.py` — the rule that follows from it

After the solve, for every node with in-degree 0 and `t > 0`, look up the
model's own probability for an edge from each solved node in the previous frame
that already continues into exactly one daughter. Take the best above a floor,
one new child per parent, capped per movie.

Two properties, both by construction:

- it **ranks by the model's probability**, not by geometry.
  `division_recovery.py` ranks by a 3.5–11 um length band because it was built
  for a pipeline that had no score for the pair. Ours has one, and
  `division_ceiling` computes both rankings side by side for exactly this
  comparison.
- it **cannot displace anything.** The orphan had in-degree 0; the parent goes
  from one child to two, which is the out-degree cap the official metric
  applies anyway. So the solver's own optimum is untouched.

### Why it is cheap — the per-fork accounting

Verified against `score_one` on a synthetic fixture (`fork_pass.selftest()`,
16 checks, both directions):

| the added fork is | edge term | division term |
|---|---|---|
| correct | **+1 TP, −1 FN** | +1 div TP |
| wrong, on an annotated cell | +1 FP | +1 div FP |
| wrong, among unannotated cells | **nothing** | **nothing** |

The third row is why this is worth running: only ~6% of predicted nodes match a
GT node at all, so most mistakes this rule can make are invisible to both
terms. That is the same leniency that stops sparse annotation from punishing
every correct link into an unannotated cell — but it is also exactly what the
2026-07-17 patch closed for the hub-node exploit. **The cap and the floor exist
so this stays a rule and not fork-spam.** Do not raise the cap to chase the
division term.

### What has NOT been measured

`fork_pass.py` has never been run on real data — no GPU session has been open
since it was written. What exists is the rule, the harness, and the accounting
proof. The sweep is stage 3 of `run_sweep.py` and costs no quota.

Read it in this order:

1. **The floor 1.01 control row.** No probability reaches it, so that row must
   reproduce the linking sweep's cell at the same threshold/disappearance. If
   it does not, stop — the harness is wrong.
2. **`adj_edge_jaccard` against that control.** 12 datasets and ~10k edges make
   this readable. If it does not fall, the pass is free.
3. **The division columns: direction only.** 12 datasets hold ~9 division
   events. `division_jaccard` off that is noise, and this log has already
   ignored that warning once.

If the pass is free on the edge term, the division count has to be settled on a
wider cache — ~66 datasets is where ~50 events live.

---
## ⭐ The scorer bug (2026-08-18)

Read from the real `tracking_cellmot/metrics.py::_evaluate_matched_graph`:

```
out_valid = the edge's SOURCE matched a GT node with out_degree > 0
in_valid  = the edge's TARGET matched a GT node with in_degree  > 0
the edge counts toward the denominator iff (out_valid OR in_valid)
```

**One matched endpoint is enough.** Our scorer required BOTH. Measured effect on
24 datasets of the classical pipeline: **FP 2 -> 289**. 99.3% of false positives
were invisible.

Three more rules we also lacked, all now implemented:

1. edges with `t_target != t_source + 1` are **discarded**, not penalised — so
   gap-closing links across 2+ frames earn nothing unless intermediate nodes are
   emitted too
2. merges collapse: one predicted edge per matched GT edge
3. out-degree capped at 2

### The consequence nobody had spotted: a missed detection costs twice

GT has `g1 -> g2`. We detect `g1` but miss `g2`. The linker connects our `g1`
to something else:

- that edge has `out_valid = True` and is not a TP -> **FP**
- the GT edge `g1 -> g2` has no match -> **FN**

The old scorer counted only the FN. Detection recall is therefore worth
substantially more than every earlier measurement suggested — which is exactly
why the official U-Net (recall 0.997) beats the DoG detector (0.70) by far more
than the recall numbers alone imply.

### Re-measured: over-detection survives the fix

Classical pipeline, 24 datasets x 15 frames, corrected scorer:

| ratio | 0.8 | 1.0 | 1.4 | 2.0 | 2.8 |
|---|---|---|---|---|---|
| adj_J | .328 | .350 | .413 | .447 | **.457** |
| N_pred/N_true | .27 | .34 | .47 | .69 | .94 |
| link_rate | .949 | .945 | .932 | .919 | .911 |

Still rising at the top of the range, so `ratio=2.0` is not the optimum. The
falling `link_rate` is the cost of over-detection finally becoming visible; the
recall gain still beats it. `N_pred/N_true` reaching only 0.94 means the DoG
detector is out of candidates and `nms_um` is the real lever.

---

## What the published artefacts revealed (2026-08-18)

### DeepCenter's split is a FACT, and it is a batch split

`weights/full_frame_center/split_manifest.json`:

```
all:   199 datasets
train:  71  = exactly the 44b6_ batch
val:   128  = exactly the 6bba_ batch
```

Cause, from `train_full_frame_center_detector.py:404-419`: the author does a
group split on `name.split("_")[0]`, which yields only **two** groups. Then
`n_val = max(1, round(2 * 0.10)) = 1` forces one whole group into validation.
Intent was right, execution collapsed on a two-group dataset.

**Consequence:** every `6bba_` dataset is guaranteed unseen by DeepCenter. That
is why the held-out pool is now `6bba`-only, and it does not depend on the
19-dataset derivation being right.

### DeepCenter's shipped checkpoint is the un-evaluated one

| checkpoint | epoch | val_loss |
|---|---|---|
| `best.pt` | 2 | 0.04500 |
| `checkpoint_last.pt` | 500 | 0.31674 |

`SNAPSHOT_MANIFEST.json` says *"Gate diagnostics are computed from best.pt"* —
so the threshold sweep, and therefore the `0.25` the 0.916 notebook uses, was
calibrated on `best.pt`. The notebook loads `checkpoint_last.pt` and asserts
`epoch == 500`. **The two checkpoints have no detection-level comparison
anywhere in the artefact.**

Do **not** read the 7x val_loss as "the detector got worse". The loss is
positive-unlabeled weighted BCE with `w_ignore=0.05` on bright unlabelled
voxels — 97% of real cells sit in that region labelled as negative, so a model
that learns "what a cell looks like" is punished for finding them, and BCE
punishes confident predictions hardest. Rising val_loss is consistent with a
*better* detector. The gate sweep is the only valid evidence, and it exists for
one checkpoint only.

### The primary model's split is still unknown

`unet_transformer/split_0/checkpoint_last.pth` carries `epoch: 402`,
`best_score: 0.9835`, `fold: 0`, `model_config` — and **no file list**. See
`docs/holdout-split.md`.

---

## Where the code lives

Local folder `D:\programming\cell tracking\`:

| File | Purpose |
|---|---|
| `viz_cells.py` | inspect / viz / gif / survey, plus all shared I/O |
| `cell_count.py` | features / fit / predict — estimate N_true from images |
| `submit_baseline.py` | predict / score / sweep / verify, **and the local scorer** |
| `check_weights.py` | inspect the published checkpoints for their training split |
| `error_analysis.py` | classify every edge error: detection / dropped / confusion / spurious |
| `division_features.py` | do geometry, intensity or elongation distinguish a dividing cell? |
| `division_recovery.py` | add missed divisions back into an ILP graph, post-hoc — GEOMETRY-ranked, superseded by `fork_pass.py` |
| `division_ceiling.py` | funnel over every GT division: how many forks could a rule recover here at all |
| `fork_pass.py` | add back the forks the SOLVER refused, ranked by the model's own edge probability |
| `division_strip_cell.py` | static filmstrip of one GT division |
| `division_gif_cell.py` | animated view of one GT division |
| `sweep_edge_threshold.py` | sweep the linker threshold offline from one inference run |
| `read_gate.py` | read DeepCenter's tuning files (threshold sweep, history) |
| `make_fake_data.py` | synthetic data for smoke tests (numbers meaningless) |
| `cache_candidates.py` | dump `(coords, src, tgt, prob, dist)` per dataset — the one GPU stage |
| `link_replay.py` | re-solve a cached candidate set on CPU; the replay sweep |
| `stranded_audit.py` | count what the 0.5 gate strands, and how much of it is real |
| `run_sweep.py` | the CPU Kaggle kernel: link sweep, then the fork sweep |
| `run_cells.md` | Kaggle cells, one fenced block per cell |
| `submission_notebook.md` | the cells that produced the 0.876 submission |

Versions as of 2026-08-20: `viz_cells`, `cell_count`, `submit_baseline`,
`error_analysis` and `sweep_edge_threshold` at `2026-08-18f`; the three
`division_*` files at `2026-08-19a`. Rule 3 requires the versions printed in a
Kaggle cell to match what you just uploaded — it does not require every file to
carry the same date. Cowork's cloud working directory is wiped
between sessions — write finished files out immediately.

---

## Running the official pipeline

```bash
git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git
pip install -e . ; pip install "tracksdata @ git+.../tracksdata@main" zarr tqdm

python scripts/predict_unet_transformer.py \
    --splits splits.json --split 0 \
    --weights .../unet_transformer/split_0/edge_predictor_best.pth \
    --det-threshold 0.99 --evaluate
```

- the package is `tracking_cellmot`; pilkwang's repackaged copy renames it
  `biohub_tracking`
- `--debug-video` takes a path with **no `.zarr` suffix** — `predict()` uses the
  path's name for the output filename, and `x.zarr.geff` will not match GT
- `predict()` **wipes every .geff in the output dir** on each call, so `--slice`
  runs do not accumulate
- `--use-ilp` needs `pyscipopt`; `--ilp-division-weight` is the direct knob on
  fork creation
- pip prints a wall of conflicts about `bigframes`, `ydata-profiling`,
  `google-colab`, `cuml`, `cudf` — none are in the pipeline

---

## The metric

`score = adjusted_edge_jaccard + 0.1 * division_jaccard`

- **`N_true`** = train geff metadata field `estimated_number_of_nodes`. Test has
  none.
- **Penalty** `max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))`, no upper
  clamp, weak coefficient.
- **Not a pure micro-average**: `adj_edge_jaccard` is per-dataset, averaged with
  weights `w_i = TP + FP + FN`. This is exactly why a flat constant caps out.
- **Edge FP rule**: an edge counts toward the denominator if **either** endpoint
  matched a GT node of the right degree. See the scorer-bug section above.
- **Division metric**: 574 lines of local-topology validation in
  `division_metrics.py`, deliberately NOT ported. Our
  `score_divisions_simple` is a lenient upper bound — trust the repo's number.

---

## Full survey of the 199 train datasets

| Metric | Value |
|---|---|
| Datasets | 199 (`44b6` 71, `6bba` 128) |
| Shape — identical everywhere | `(100, 64, 256, 256)` uint16; voxel scale uniform |
| GT nodes / edges | 133,318 / **128,883** |
| Divisions | **151** (1 per 854 edges; 56% of datasets have none) |
| N_true total | 4,725,117 -> **GT labels only 2.82% of it** |
| True cells/frame | min 38 / median 179 / max 786 (**20x spread**) |
| Pooled displacement | p50 1.82 / p99 6.71 / p100 60.76 um |

### By imaging batch

| | `44b6` (71) | `6bba` (128) |
|---|---|---|
| **Share of score weight** | 15% | **85%** |
| True cells/frame (median) | 327 | **97** |
| Annotated fraction (median) | 0.77% | 9.71% |
| Displacement p99 (median) | 4.90 um | **7.81 um** |

- `corr(annotation density, true density) = -0.20` — **negative**. N_true cannot
  be inferred from the GT; hence `cell_count.py`
- `corr(true density, displacement p99) = -0.39` — denser videos move less
- **43 of 199 datasets have GT that does not span all 100 frames.** The
  node-count penalty counts the whole movie, so **detection must run on every
  frame**

---

## Submission mechanics

**Code Competition, hidden test set.**

- Notebook submissions only; **12 h limit**; **internet disabled**; output must
  be named exactly `submission.csv`
- The visible `test/` holds **4 datasets that are copies of train files**
  (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` —
  quantile tables match digit for digit). Real data is swapped in at rescore
- **Never hardcode dataset count or names**
- Those 4 datasets have ground truth under train, so the real submission CSV can
  be scored — `submit_baseline.py verify`

### Runtime budget — RESOLVED 2026-08-20, and the margin is large

Measured from the scored submission's own cell execution timestamps (notebook
`faustine649/cell-tracking-during-development`, 4 visible datasets,
`predict_video` + ILP, T4):

```
install wheels          26 s
import + load model     11 s
inference, 4 datasets  377 s   ->  94.2 s/dataset
```

Extrapolated to the ~199-volume hidden test: **~5.2 h of the 12 h limit.**
Corroborated independently — forum reports put the visible->submission runtime
ratio at 40-50x, and 377 s x 45 = 4.7 h.

The earlier "~5 min/dataset, so 12 h allows only ~144 datasets" estimate was
wrong by 3x. **There is roughly 2.3x of inference budget spare** — enough for
TTA, a wider candidate-edge search, or both.

One residual risk: the notebook falls back to greedy linking if `pyscipopt`
fails to import. It warns (rule 1), but that warning only appears in the
rescore log. Confirm ILP actually ran before reading 0.876 as the ILP+A number.

### Offline dependencies

The classical path needs no installs: zarr v3 chunks are read directly via
`blosc2` (ships with the Kaggle image; `zarr` does not). Both `open_image()` and
`read_geff()` have zarr-free fallbacks, verified byte-identical.

**The official pipeline is a different problem** — it needs `torch`,
`tracksdata`, `zarr`, `rustworkx`. Those must come from an offline wheel
dataset; pilkwang's support pack ships 62 wheels and a
`kaggle_dependency_install_command.txt` for exactly this.

---

## Timeline

| Date | Event | Status |
|---|---|---|
| **2026-09-22** | Team merger / new-entrant deadline | confirmed via the API |
| **2026-09-29** | **Final submission** | UTC+8: 09-30 07:59 |

40 days left as of 2026-08-20. Last weekend (09-26/27) is for submitting and
writing up only.

---

## To do

### Immediate — all on the official pipeline


- [x] ~~Confirm ILP+A on all 12 held-out datasets~~ — done 2026-08-19,
      **0.9269**, improved 12 of 12.
- [x] ~~SUBMIT~~ — done 2026-08-19, **LB 0.876**. Runtime fits comfortably
      inside 12 h, so that risk is closed. Offline wheels work; do NOT install
      every wheel in the support pack, it clobbers numpy (see
      `submission_notebook.md`).
- [ ] **Explain the 0.051 offset.** The old 0.916-notebook probe is dead — that
      badge is pre-patch, see `docs/public-solutions-recon.md`. The live probe
      is a **`44b6`-only holdout**: every number we own comes from `6bba`, which
      carries a 7.47% duplicate-frame rate against `44b6`'s zero. One inference
      pass, and it closes a real blind spot.
- [ ] **Measure the field-of-view drift** — [CPU, GT only]. The movies visibly
      pan, so cells enter and leave the volume. Per frame pair, take the MEDIAN
      DISPLACEMENT VECTOR of GT cells (not the scalar): coherent drift gives a
      large vector, independent motion cancels to ~0. Cumulative drift over 99
      frames against the 104 um box decides whether population growth can
      estimate the division count at all, and gives the appearance-based
      division recovery its sharpest feature.
- [ ] **`--ilp-division-weight` between 1.0 and 0.3.** A fork is taken iff
      `edge_prob > division_weight - appearance_weight`, so 1.0/0.0 forbids all
      forks and 0.3/0.0 floods. 0.85 would fork only on very confident second
      edges. Untested.
- [ ] **`--det-threshold` sweep** `{0.99, 0.995, 0.999}`. More predicted nodes
      correlates with lower edge J and more spurious forks; the node-count
      penalty alone costs 0.014.
- [ ] **Look at `6bba_7af54fde`** — only 97 cells/frame yet J=0.828 and the
      worst over-detection (1.28). Density does not explain it; motion probably
      does. One `run_gif` plus its displacement percentiles would say.
- [x] ~~Re-run the 0.916 notebook on these same 12 datasets~~ — **dropped
      2026-08-20**: the badge is pre-patch, so that comparison measures the
      closed exploit, not the post-processing stack.
- [x] ~~Settle the hidden test size / runtime budget~~ — done 2026-08-20:
      ~199-200 volumes, ~5.2 h of 12 h. See the runtime section above.

### Deferred

- [ ] **The fork sweep** — stage 3 of `run_sweep.py`, written and self-tested,
      never run on data. **Deliberately parked behind the edge/node work** per
      the 2026-08-21 priority ruling; `RUN_FORK_SWEEP` is off by default. When
      it does run: read the floor 1.01 control first, then `adj_edge_jaccard`,
      then the division columns as a direction only.
- [ ] `cell_count.py` on real data — only matters if the classical path revives,
      or if N_true estimation helps the official pipeline's node-count penalty
- [ ] Train the U-Net ourselves, with a split we choose

---

## Historical: the classical DoG pipeline

Superseded, kept because the reasoning still applies to the node-count penalty.

**Submitted config (public LB 0.681, rank 1940/2470, 2026-08-17):**

```
sigma_um=2.0  sigma_ratio=1.6  nms_um=3.0  gate_um=10.0
per_frame=100  ratio=2.0  max_gap=1  prune_isolated=True  truncate=3.0
```

Runtime 92 s/dataset, single core, 100 frames.

Findings that still hold:

1. **Under-detecting is catastrophic, over-detecting is cheap.** One dataset at
   2.92x N_true scored J=0.84; another at 0.28x scored J=0.18.
2. **Unlinked nodes are pure cost** — they consume the node-count penalty and
   can never make a TP. Pruning raised adj_J 0.6743 -> 0.6890 and TP went *up*,
   because one-to-one matching freed GT nodes for linked predictions.
3. **`node_recall` is a misleading target** — pruning lowered it while raising
   the score. Watch TP.
4. **`ratio` and `nms_um` are coupled.** `nms_um=3.5` merged neighbouring nuclei
   (they sit ~7 um apart), saturating the candidate set, so `ratio` could not
   deliver what it asked for.
5. **A per-dataset count estimate is worth +0.155** over the best flat constant.
   A flat `per_frame=300` produced essentially the same edges as oracle counts
   (TP 1470 vs 1468) but scored 0.4926 vs 0.6890 — the whole gap is the
   node-count penalty being mis-distributed across datasets.

All adj_J numbers in this section were produced by the pre-2026-08-18 scorer and
are **optimistic**. Only the directions are reliable.

---

## Traps already hit

1. **napari is unusable** — desktop GUI, Kaggle is headless. Use matplotlib MIP.
2. **Kaggle has no zarr**, and submissions have no internet.
3. **OME-Zarr is a multiscale pyramid** — `zarr.open()` returns a group.
4. **argparse dies in notebooks** — use `parse_known_args()`.
5. **matplotlib's default interpolation blurs microscopy** — `nearest`.
6. **Full-Z projection is mush** — 64 planes x 1.625 um is ~15 nuclei deep.
7. **matplotlib's default font has no CJK glyphs** — figure text in English.
8. **Cowork's cloud working directory does not persist across sessions.**
9. **Folder access is fixed at session start.**
10. **`--limit` taking the first N covers one imaging batch only** — use a
    stride. Runs with different `limit` are NOT comparable.
11. **Kaggle Dataset caching plus `sys.modules` caching** runs stale code
    **without raising**. Hence `SCRIPT_VERSION` in every file.
12. **Cross-file imports of private helpers** break when the Dataset copy is a
    version behind.
13. **Too much output kills a Kaggle cell** — never print per dataset in a loop.
14. **Recomputing detection inside a sweep** — caching it made sweeps 4.5x
    faster with byte-identical results.
15. **Parameters interact** — `ratio` had to be re-swept after `nms_um` changed.
16. **`!pip install` with internet disabled spins for ~10 minutes** retrying
    before failing.
17. **A `.geff` IS a zarr store.** "The geff *library* is optional" is not the
    same as "zarr is optional".
18. **`node_recall` is a misleading target.**
19. **Reimplementing an official metric is a liability.** Ours was wrong for
    weeks in the flattering direction, and every tuning decision rode on it.
    Fixed and then cross-checked against the official scorer on the same 12
    predicted graphs: edge TP/FP/FN identical, adj_J agreeing to 5 decimals.
    Do that cross-check after every scorer change; it is cheap and it is the
    only thing standing between a fast local sweep and weeks of wasted work.
20. **`Path("/kaggle/input").rglob("*")` appears to hang** — the competition
    mount holds millions of zarr chunk files. Prune `.zarr` trees.
21. **A silent cell is not a hung cell.** Some scripts print once at the start
    and once at the end; check the GPU meter before killing anything.

---

*Last updated: 2026-08-21, after `fork_pass.py` — written and self-tested, not yet run on data*
