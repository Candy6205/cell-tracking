# Cell tracking — Kaggle cells

One fenced block = one Kaggle cell. Everything is a code block, including the
shell commands — that is the whole point of using Markdown here instead of a
`.py` file, which cannot hold `!git clone` or `%cd` without breaking.

Each section is tagged **[CPU]** or **[GPU]**. Run section 1 first, always.

---

## 1. Version check — [CPU], seconds

Kaggle Dataset caching serves stale code **without raising anything**. All three
must print `2026-08-18f`. A mismatch means the Dataset was not updated — it is
not a bug in the code.

```python
import importlib, sys

CODE = "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking"
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import viz_cells, cell_count, submit_baseline as sb
for m in (viz_cells, cell_count, sb):
    importlib.reload(m)
    print(m.__name__, m.SCRIPT_VERSION)
```

---

## 2. Re-baseline the local scorer — [CPU], ~2 min

`score_one` was rewritten on 2026-08-18 against the real
`tracking_cellmot/metrics.py`. The old false-positive rule required **both**
endpoints of a predicted edge to match ground truth before the edge could count
as an FP; the official rule needs only **one**.

Every local number produced before today is optimistic by an unknown amount.

Watch the line directly under `edge TP/FP/FN` — it reports how many FP the old
scorer was throwing away.

```python
sb.run_score(limit=24, max_frames=15)
```

---

## 3. Re-sweep `ratio` — [CPU], ~5 min

`--ratio 2.0` (over-detect to 2x the estimated cell count) was chosen with the
broken scorer, which could not see the false positives that over-detection
creates. If the optimum moves down, the submitted 0.681 was tuned against a
metric that does not exist.

```python
sb.run_sweep(param="ratio", values=[0.8, 1.0, 1.4, 2.0, 2.8],
             limit=24, max_frames=15)
```

**Result, 2026-08-18** — over-detection survives the fix, but 2.0 is not the
optimum; adj_J was still rising at the top of the range:

| ratio | 0.8 | 1.0 | 1.4 | 2.0 | 2.8 |
|---|---|---|---|---|---|
| adj_J | .328 | .350 | .413 | .447 | .457 |
| N_pred/N_true | .27 | .34 | .47 | .69 | .94 |
| link_rate | .949 | .945 | .932 | .919 | .911 |

`link_rate` falling is the cost of over-detection finally becoming visible —
the old scorer could not see it. It just loses to the recall gain.

`N_pred/N_true` reaching only 0.94 at ratio 2.8 means the DoG detector has run
out of candidates. Past that point `ratio` does nothing and **`nms_um` is the
real lever** — it controls how many maxima exist at all.

### 3b. Extend the sweep — [CPU], ~6 min

```python
sb.run_sweep(param="ratio", values=[2.8, 4.0, 6.0],
             limit=24, max_frames=15)
```

### 3c. Then the knob that actually creates candidates — [CPU], ~8 min

Current default is `nms_um=3.5`. Smaller = more maxima survive = more
candidates for `ratio` to spend.

```python
sb.run_sweep(param="nms_um", values=[2.0, 2.5, 3.0, 3.5],
             limit=24, max_frames=15, ratio=4.0)
```

---

## 4. The bare official pipeline — [GPU]

No DeepCenter, no second seed, none of the `BIOHUB_*` post-processing. Just the
official repo plus the public weights.

**Turn internet ON** (Settings -> Internet) for this section.

### 4a. Setup — [GPU], ~4 min first time, ~2 min after

On a returning session only the `pip install` matters; everything else no-ops.

Everything a fresh session needs: clone, every `pip install`, paths, both split
files, and the `--edge-threshold` patch. Idempotent — safe to re-run.

With Kaggle's file persistence on, `/kaggle/working` survives between sessions —
so the clone, the splits and the patch are all still there and their steps
no-op. The **Python environment** does not persist, which is the only reason
`zarr` disappears. On a returning session just re-run the `pip install` part.

**Internet must be ON** (Settings -> Internet).

```python
import json, pathlib, subprocess, sys

REPO = pathlib.Path("/kaggle/working/repo")
if not (REPO / "pyproject.toml").exists():
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
                    str(REPO)], check=True)

PKGS = [
    "-e", str(REPO),                                    # tracking_cellmot itself
    "tracksdata @ git+https://github.com/royerlab/tracksdata@main",
    "zarr>=3.0.10",      # OME-Zarr images AND .geff stores
    "geff",              # GeffMetadata, used by evaluate.py
    "polars",            # tracksdata's dataframe layer
    "rustworkx",         # tracksdata's graph backend
    "scikit-image",
    "tqdm",
    "pyscipopt",         # required by --use-ilp
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + PKGS, check=True)

TRAIN   = pathlib.Path("/kaggle/input/competitions/"
                       "biohub-cell-tracking-during-development/train")
WEIGHTS = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                       "biohub-tracking-support-pack-50ep-v1/"
                       "weights/unet_transformer/split_0/edge_predictor_best.pth")
PRED    = REPO / "predictions/unknown/unet_transformer/split_0"
assert TRAIN.exists(),   TRAIN
assert WEIGHTS.exists(), WEIGHTS

# The 12 held-out 6bba_ datasets. DeepCenter's split_manifest.json proves the
# whole 6bba batch was held out of ITS training -- a fact, unlike our derived
# 19-dataset holdout.
TWELVE = ["6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1", "6bba_3a1849c2",
          "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
          "6bba_7b5d3b2c", "6bba_aeee7805", "6bba_afb141ff", "6bba_c27cba08"]
THREE  = ["6bba_3abfe10a",   # 519 true cells/frame -- worst edge J
          "6bba_2312ac41",   # 119 -- best
          "6bba_aeee7805"]   # 50  -- sparse control
(REPO / "splits.json").write_text(json.dumps([{"train": [], "test": TWELVE}]))
(REPO / "splits3.json").write_text(json.dumps([{"train": [], "test": THREE}]))

# `threshold` (the linker's edge threshold) is hardcoded in PredictConfig with
# no CLI argument. Add one.
ps = REPO / "scripts/predict_unet_transformer.py"
src = ps.read_text()
if "--edge-threshold" not in src:
    src = src.replace('    parser.add_argument("--use-ilp", action="store_true",',
                      '    parser.add_argument("--edge-threshold", type=float, default=0.5)\n'
                      '    parser.add_argument("--use-ilp", action="store_true",')
    src = src.replace("    cfg = PredictConfig(\n        det_threshold=args.det_threshold,",
                      "    cfg = PredictConfig(\n        threshold=args.edge_threshold,\n"
                      "        det_threshold=args.det_threshold,")
    ps.write_text(src)

print("repo         :", REPO.exists())
print("edge-threshold patch:", "--edge-threshold" in ps.read_text())
print("splits       :", len(TWELVE), "and", len(THREE), "datasets")
print("predictions  :", PRED)
```

### 4b. Verify the install — seconds

`pip` prints a wall of red about `bigframes`, `ydata-profiling`, `google-colab`,
`cuml`, `cudf`. **None of those are in our pipeline** — ignore them. The one
upgrade that can genuinely break things is `numpy` jumping a major version under
`torch`, so verify rather than guess.

```python
import importlib
for mod in ("numpy", "torch", "zarr", "geff", "polars", "rustworkx",
            "tracksdata", "tracking_cellmot", "pyscipopt"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:20s} {getattr(m, '__version__', 'n/a')}")
    except Exception as e:
        print(f"{mod:20s} FAILED: {type(e).__name__}: {e}")

import torch
print("cuda available :", torch.cuda.is_available())
print("device         :", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "CPU ONLY -- switch the accelerator on")
```

If anything says `FAILED`, restart the kernel once (Run -> Restart) and re-run
**this cell only** — the packages are already on disk. Do not re-run 4a.

### 4c. Smoke test on ONE dataset — ~5 min

Confirms the weights load, the geff writes, and `--evaluate` prints the official
metric — before spending an hour on twelve.

The path has **no `.zarr` suffix** on purpose: `predict()` uses the path's name
for the output filename, and `x.zarr.geff` would not match the ground truth
`x.geff`.

`--det-threshold 0.99` is the repo default. The 0.916 notebook uses 0.96875;
that is one of the few knobs worth sweeping later.

```python
%cd /kaggle/working/repo
!python scripts/predict_unet_transformer.py \
    --debug-video {TRAIN}/6bba_2312ac41 \
    --weights {WEIGHTS} \
    --det-threshold 0.99 \
    --evaluate
```

### 4d. The real run — 12 datasets, ~1 h

**This is THE baseline number.** Everything after it is measured as a delta
against it.

`--evaluate` prints, per dataset and then overall:

```
edge TP/FP/FN, div TP/FP/FN, n_pred
score = adj_edge_jaccard + 0.1 * division_jaccard   <- LB-comparable
```

The public leaderboard is 0.916 **with** all the extras. The gap between this
number and 0.916 is exactly what the post-processing pile is worth.

**It runs SILENTLY.** `predict()` prints one line at the start, then nothing
until all twelve are done — the per-dataset `tqdm` bar is disabled whenever
stderr is not a tty, which it never is under `!python` in a notebook. A cell
that has shown nothing for forty minutes is almost certainly working, not hung.

To tell working from hung: the Kaggle sidebar's GPU meter should be pegged, and
the cell timer keeps counting. If GPU sits at 0% for minutes, something is
wrong.

Run it in slices of three instead, so there is feedback every ~15 min and a
crash costs one slice rather than the whole hour. `--slice` indexes into the
test list from `splits.json`.

```python
%cd /kaggle/working/repo
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits.json --split 0 \
    --weights {WEIGHTS} --det-threshold 0.99 --slice :3
```

```python
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits.json --split 0 \
    --weights {WEIGHTS} --det-threshold 0.99 --slice 3:6
```

```python
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits.json --split 0 \
    --weights {WEIGHTS} --det-threshold 0.99 --slice 6:9
```

```python
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits.json --split 0 \
    --weights {WEIGHTS} --det-threshold 0.99 --slice 9:
```

WARNING: `predict()` **deletes every existing .geff in the output dir** at the
start of each call. So the slices above do NOT accumulate — each one wipes the
previous. Score each slice separately with the cell below, or run the single
un-sliced command if you want one combined number.

Score whatever is currently in the predictions dir (CPU, seconds):

```python
!python scripts/evaluate.py \
    --pred-dir predictions/unknown/unet_transformer/split_0 \
    --gt-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train
```

Or, for one combined number over all twelve in a single go (~1 h, silent):

```python
%cd /kaggle/working/repo
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits.json --split 0 \
    --weights {WEIGHTS} \
    --det-threshold 0.99 \
    --evaluate
```

---

## 5. Cross-check the two scorers — [CPU], ~2 min

`score_one` is our reimplementation; the repo's is the real thing. Running both
on the **same** predicted graphs is the only way to know our fast local scorer
can be trusted for sweeps.

`adj_edge_jaccard` from both should agree to ~0.001. If it does not, our scorer
is still wrong and no local sweep means anything.

One known difference remains even if the edge term matches: our
`division_jaccard` is `score_divisions_simple`, a lenient approximation. The
repo's `division_metrics.py` is 574 lines of local-topology validation we have
not ported. **Trust the repo's division number, not ours.**

```python
import sys, pathlib
sys.path.insert(0, "/kaggle/working/repo/scripts")
from geffs_to_csv import geffs_to_csv

PRED = next(pathlib.Path("/kaggle/working/repo/predictions")
            .glob("*/unet_transformer/split_0"))
geffs_to_csv(PRED, "/kaggle/working/official_pred.csv")

sb.run_verify("/kaggle/working/official_pred.csv", gt_split="train")
```

---

## 5b. Where do the 1136 edge errors actually come from? — [CPU], ~1 min

Correlating twelve dataset-level scores against density is weak evidence. This
opens every one of the 10313 ground-truth edges and classifies what happened:

| bucket | meaning |
|---|---|
| `TP` | linked correctly |
| `DETECTION` | an endpoint was never detected |
| `DROPPED` | source detected, but no forward edge was emitted from it |
| `CONFUSION` | linked the source to a **different annotated cell** |
| `SPURIOUS` | linked the source to a node matching **no** GT cell |

For each bucket it reports the GT displacement and the local crowding, against
the `TP` distribution as control — so "motion" and "crowding" stop being
hypotheses and become measured failure rates per bucket.

Requires `official_pred.csv` from section 5.

```python
import importlib, error_analysis
importlib.reload(error_analysis)
error_analysis.run("/kaggle/working/official_pred.csv")
```

It also writes `/kaggle/working/error_summary.txt` (the report) and
`/kaggle/working/error_edges.csv` (one row per GT edge). **Send those files
rather than pasting output.** `run()` deliberately returns only a small count
dict — returning the per-edge list would make the notebook print all 10313
records.

Read the two rate tables at the bottom first. If the failure rate climbs
steeply with crowding but is flat in displacement, the linker needs better
discrimination between neighbours. If it climbs with displacement, the gate
radius or a motion model is the answer. If `SPURIOUS` dominates, raising
`--det-threshold` is the cheap fix. They are different projects; the tables say
which one.

---

## 5c. Edge-threshold sweep — ONE GPU run, then free

This is the highest-value experiment currently identified. `error_analysis`
attributes **60% of all edge failures** to `DROPPED`: the source and target were
both detected, and the linker emitted no edge at all. The mechanism is in
`predict_unet_transformer.py`:

```python
probs = torch.softmax(raw, dim=0)                  # normalised over SOURCES
candidates = [... if probs[i, j] > cfg.threshold]  # cfg.threshold = 0.5, hardcoded
```

For a given target, the probabilities over all candidate sources sum to 1. One
candidate gives ~1.0; two give ~0.5 each — right on the knife edge; three give
~0.33 each, and **nothing passes, so the target gets no parent**. That is
exactly the measured failure rate: 2.5% with one neighbour, 20.1% with two,
33.0% with three.

### 5c-1. Patch in a `--edge-threshold` flag

Already applied by 4a. Kept here for reference; re-running it is harmless:

```python
import pathlib
p = pathlib.Path("/kaggle/working/repo/scripts/predict_unet_transformer.py")
s = p.read_text()
if "--edge-threshold" not in s:
    s = s.replace('    parser.add_argument("--use-ilp", action="store_true",',
                  '    parser.add_argument("--edge-threshold", type=float, default=0.5)\n'
                  '    parser.add_argument("--use-ilp", action="store_true",')
    s = s.replace("    cfg = PredictConfig(\n        det_threshold=args.det_threshold,",
                  "    cfg = PredictConfig(\n        threshold=args.edge_threshold,\n"
                  "        det_threshold=args.det_threshold,")
    p.write_text(s)
print("patched:", "--edge-threshold" in p.read_text())
```

### 5c-2. Three datasets at a LOW threshold — [GPU], ~15 min

Dense / medium / sparse, so the crowding effect is visible.

Run inference **once** at 0.05. Because the greedy assignment processes
candidates in descending probability order, lowering the threshold only appends
to the tail of that list — so the edge set at any higher `T` is exactly the
subset of this run with `prob > T`. One run contains every threshold.

```python
import json, pathlib
REPO = pathlib.Path("/kaggle/working/repo")
(REPO / "splits3.json").write_text(json.dumps([{"train": [], "test": [
    "6bba_3abfe10a",   # 519 true cells/frame -- worst edge J
    "6bba_2312ac41",   # 119 -- best
    "6bba_aeee7805",   # 50 -- sparse control
]}]))
print("ok")
```

```python
%cd /kaggle/working/repo
!python scripts/predict_unet_transformer.py \
    --splits /kaggle/working/repo/splits3.json --split 0 \
    --weights {WEIGHTS} --det-threshold 0.99 --edge-threshold 0.05
```

### 5c-3. Sweep on CPU — free, seconds

```python
import importlib, sweep_edge_threshold
importlib.reload(sweep_edge_threshold)
sweep_edge_threshold.run(
    "/kaggle/working/repo/predictions/unknown/unet_transformer/split_0")
```

**Sanity check before reading anything else:** the `T=0.50` row must reproduce
the baseline for those three datasets.

Writes `edge_threshold_sweep.csv`; returns a 5-key dict.

### Result, 2026-08-18 — lowering the threshold LOSES

Self-check passed exactly: at `T=0.50`, TP/FP/FN = 2686/164/149, which is
816+1113+757 / 124+26+14 / 119+19+11 from the 12-dataset run, digit for digit.
The subset argument is validated.

| T | TP | FP | FN | adj_J |
|---|---|---|---|---|
| **0.50** | 2686 | 164 | 149 | **0.8826** |
| 0.40 | 2719 | 215 | 116 | 0.8785 |
| 0.30 | 2726 | 254 | 109 | 0.8696 |
| 0.05 | 2728 | 338 | 107 | 0.8472 |

Each recovered true edge costs about **four** new false ones. Jaccard cannot pay
that exchange rate.

Two things this rules out and one it points at:

- the `DROPPED` failures are **not** mostly threshold-limited. TP saturates by
  `T=0.15`; **107 of the 149 FN are unreachable at any threshold.**
- those 107 are blocked by the hard cap `max_parents_per_node = 1`
  (first-come-first-served on each target), which no threshold change touches
- the FP explosion comes from `max_children_per_node = 2`: at a low threshold
  every source emits a second edge, which is nearly always wrong **and** turns
  that node into a spurious fork

So the remaining lever for `DROPPED` is **global assignment (`--use-ilp`)**,
not the threshold.

### 5c-4. Is the second child worth anything? — [CPU], free

`sweep_edge_threshold` (2026-08-18d) also scores a **one-child-per-source**
variant of every row, as an extra column. It costs nothing — same predicted
graphs, one more filter.

Caveat: it is a post-hoc filter, not a re-run of the greedy assignment with
`max_children=1`, so freed target capacity is not redistributed. It is a LOWER
bound on the real setting.

Reading it: one child per source means **no forks at all**, so
`division_jaccard` would drop to 0. That currently contributes
`0.1 x 0.0265 = 0.0027`. So the variant only wins overall if its `adj_J` beats
the baseline by **more than +0.003**.

**Result, 2026-08-18:** best one-child row is `T=0.40` -> adj_J **0.8873**
(TP 2684, FP **146**, FN 151) against the 0.8826 baseline. +0.0047, almost
entirely from 18 fewer false positives. Net of the lost division term that is
**+0.0020** — real, but small, and it throws away all 12 genuine divisions to
kill 214 spurious ones.

### 5c-5. A separate threshold for the SECOND child — [CPU], ~3 min, free

The real defect is not that a source *can* have two children; it is that both
outgoing edges share one threshold, so any source clearing 0.5 twice becomes a
fork. A higher bar for the second edge only keeps real divisions and drops the
junk — and it attacks the 214 spurious forks directly, which is where the
division term lives.

Grid over (edge threshold `T`, second-child threshold `T2`). `T2 = never` is
the one-child variant from 5c-4.

```python
import importlib, sweep_edge_threshold
importlib.reload(sweep_edge_threshold)
sweep_edge_threshold.run_fork(
    "/kaggle/working/repo/predictions/unknown/unet_transformer/split_0")
```

It reports `score_full = adj_J + 0.1*div_J`, not `adj_J` — this knob trades the
two terms against each other, so `adj_J` alone would mislead.

**Caveat:** the division term here is `score_divisions_simple`, a lenient
approximation of the official 574-line metric. Use the grid to pick a
direction, then confirm the winner with the official `evaluate.py`.

Writes `fork_threshold_grid.csv`; returns a 4-key dict.

**Always pass `expect=`.** `predict()` wipes every `.geff` in its output dir on
each call, so reading that dir while a run is in progress silently scores a
partial set — which happened once and produced a table that looked fine and
compared nothing:

```python
THREE = ["6bba_3abfe10a", "6bba_2312ac41", "6bba_aeee7805"]
sweep_edge_threshold.run_fork(PRED_DIR, expect=THREE)
```

Since 2026-08-18f both entry points print the dataset names, warn when the set
differs from `expect`, report the fork count, and warn when every edge shares
one probability — all three of which make a degenerate grid obvious instead of
silent.

---

## 6. Extras — [CPU]

### Look at a movie

Dense datasets need an ROI crop — 700 cells in a 256x256 frame is unreadable.
Red circles are ground-truth-annotated cells.

```python
viz_cells.run_gif(dataset="44b6_706092f0", roi="64,192,64,192",   # 699 cells/frame
                  slab=9, step=2, fps=8, out="dense.gif")
viz_cells.run_gif(dataset="6bba_2312ac41", roi="64,192,64,192",   # 120 cells/frame
                  slab=9, step=2, fps=8, out="sparse.gif")

from IPython.display import Image
Image("dense.gif")
```

### Inspect the published weight artefacts

Already run once. Attach `pilkwang/biohub-temporal-unet3d-seed314159-v1` and
re-run to learn the second seed's split too.

```python
exec(open(f"{CODE}/check_weights.py").read())
```

```python
exec(open(f"{CODE}/read_gate.py").read())
```
