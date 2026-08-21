"""
cache_det_sweep.py -- where should `det-threshold` go?

Why this is the indicated experiment
------------------------------------
`stranded_audit.py` counted 22,474 stranded targets that match a GT node only
1.5% of the time against 6.1% for targets with a confident parent. Most of them
are detections the model invented. They are charged twice: they pay the
node-count penalty, and they bury the 278 genuinely recoverable edges at a
ratio of 1.2%, which is what makes lowering the edge gate a coin flip.

So shrinking the spurious pool is not just a penalty saving. It is the
precondition that makes the edge-threshold lever safe.

Why it cannot be replayed from the existing cache
-------------------------------------------------
The cache stores node coordinates, not detection confidences. Changing
`det_threshold` changes which peaks become nodes at all, so it needs a fresh
inference pass per value. Everything downstream still replays on CPU.

What to read in the output
--------------------------
This run only CACHES and counts; it does not score. That was the flaw in its
first version, and the counts alone pointed the wrong way. The caches it writes
to `/kaggle/working/cache_det/det{v}/` are the deliverable -- score them with
`link_replay` at the converged linking optimum before concluding anything.

What the counts are still good for: `nodes` should RISE as the threshold falls,
and the question is whether `recoverable` falls with it. Lowering the gate only
helps if the extra detections are the endpoints of GT edges we currently cannot
reach.

`recoverable` is the count of GT edges whose target is detected but stranded,
i.e. the 278 figure recomputed per threshold. It must not collapse: cutting the
spurious pool by throwing away the recoverable edges too is not progress.

Note the direction. `docs/public-solutions-recon.md` item 3 proposed sweeping
det-threshold DOWN to 0.95, copied from the public tier, and the first version
of this file argued UP from the stranded-pool audit. Scoring the converged
linking optimum settled it: recon was right and the audit argument was not,
because it counted nodes and never scored them.

    kaggle kernels push -p .
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-20a"

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# CORRECTED 2026-08-20. The first run of this file swept UPWARD (0.995, 0.999)
# on the argument that the stranded pool was too dirty. It counted nodes and
# never scored them, which was the mistake.
#
# Scored, the picture inverts. At the converged linking optimum
# (threshold 0.40, disappearance 3.5) the remaining loss splits:
#
#     detect_ceiling 0.9825  ->  180 of the 343 FN are edges whose endpoint
#                                was never detected              52%
#     link_rate      0.9839  ->  163 are reachable but unlinked  48%
#
# Detection is the larger half now. progress.md still says it contributes
# 0.5% and that every hour spent on it is wasted; that was measured at the
# old configuration, FN 450 and link_rate 0.9530, and no longer holds.
#
# The threshold/disappearance pair only MOVES loss between the two. Raising
# disappearance buys link_rate by spending detect_ceiling: 0.9919/0.9775 at
# 0.35/1.5 against 0.9773/0.9859 at 0.35/5.5. To go further the ceiling has
# to rise, which means detecting MORE, not less.
#
# Which is what /733877 measured on hand-built graphs: an unmatched predicted
# node is never a false positive, the whole cost is 0.1 x over-prediction,
# and thresholds in the field are too high. Ours is 0.99.
#
# 0.99 is the submitted value and is already cached by cache_candidates.py,
# so it is not repeated here. DOWNWARD now: the ceiling is what binds.
DET_THRESHOLDS = (0.98, 0.95)

CAND_THRESHOLD = 0.05
TOPK = 16

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    entries = sorted(e.name for e in os.scandir("/kaggle/input")) \
        if pathlib.Path("/kaggle/input").exists() else []
    raise SystemExit(f"support pack not attached; /kaggle/input holds {entries}")

KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
staged = sorted(WH.glob("*.whl"))
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    + [str(w) for w in staged], capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
shutil.rmtree(WH, ignore_errors=True)

import torch                                                  # noqa: E402

for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))
import predict_unet_transformer as P                          # noqa: E402

CODE = None
for c in ("/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
          "/kaggle/input/datasets/faustine649/cell-tracking"):
    if pathlib.Path(c).exists():
        CODE = c
        break
assert CODE, "code dataset not attached"
sys.path.insert(0, CODE)

from viz_cells import geff_path, read_geff, read_scale, zarr_path  # noqa: E402
from submit_baseline import _match_nodes_per_frame                 # noqa: E402

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS
COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
assert TRAIN.exists(), TRAIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("[degraded] no GPU")
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
if cap[0] < 7:
    raise SystemExit(f"[degraded] sm_{cap[0]}{cap[1]} < sm_70; request a T4")

model, window_size, downsample = P.load_model(WEIGHTS, device)

OUT = pathlib.Path("/kaggle/working/cache_det")
OUT.mkdir(exist_ok=True)

print(f"\n{'det':>7}{'dataset':<20}{'nodes':>8}{'targets':>9}{'stranded':>10}"
      f"{'str_frac':>10}{'recov':>7}{'s':>5}")

summary = []
for det in DET_THRESHOLDS:
    cfg = P.PredictConfig(det_threshold=det, use_ilp=True,
                          threshold=CAND_THRESHOLD)
    d_out = OUT / f"det{det}"
    d_out.mkdir(exist_ok=True)
    tot = {"nodes": 0, "targets": 0, "stranded": 0, "recov": 0, "gt": 0}

    for name in HOLDOUT_6BBA:
        t0 = time.time()
        try:
            coords, edges = P.predict_video(
                model, TRAIN / name, device, cfg=cfg,
                window_size=window_size, unet_batch_size=4,
                downsample=downsample)
        except Exception as exc:                              # noqa: BLE001
            print(f"[degraded] det={det} {name} FAILED "
                  f"({type(exc).__name__}: {exc}); this det value's totals "
                  f"below are missing one dataset and are NOT comparable to "
                  f"the others", flush=True)
            continue

        coords = np.asarray(coords, dtype=np.int32)
        e = np.asarray(edges, dtype=np.float64)
        if len(e) == 0:
            print(f"[degraded] det={det} {name}: zero candidates", flush=True)
            continue

        src = e[:, 0].astype(np.int32)
        tgt = e[:, 1].astype(np.int32)
        prob = e[:, 2].astype(np.float32)
        dist = e[:, 3].astype(np.float32)
        order = np.lexsort((-prob, tgt))
        ts = tgt[order]
        first = np.r_[0, np.flatnonzero(np.diff(ts)) + 1]
        sizes = np.diff(np.r_[first, len(ts)])
        rank = np.arange(len(ts)) - np.repeat(first, sizes)
        keep = order[rank < TOPK]
        np.savez_compressed(d_out / f"{name}.npz", coords=coords,
                            src=src[keep], tgt=tgt[keep], prob=prob[keep],
                            dist=dist[keep])

        has_any = np.zeros(len(coords), dtype=bool)
        has_conf = np.zeros(len(coords), dtype=bool)
        has_any[tgt] = True
        has_conf[tgt[prob > 0.5]] = True
        stranded = has_any & ~has_conf

        scale = np.asarray(read_scale(zarr_path(COMP, "train", name)),
                           dtype=float)
        gt = read_geff(geff_path(COMP, "train", name))
        matched = _match_nodes_per_frame(
            coords[:, 0].astype(int), coords[:, 1:].astype(float) * scale,
            gt["t"].astype(int),
            np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * scale)
        gt_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
        pred_of_gt = {g: p for p, g in matched.items()}
        recov = n_gt = 0
        for a, b in gt["edges"]:
            rb = gt_row.get(int(b))
            if rb is None:
                continue
            n_gt += 1
            pb = pred_of_gt.get(rb)
            if pb is not None and stranded[pb]:
                recov += 1

        n_t = int(has_any.sum())
        n_s = int(stranded.sum())
        tot["nodes"] += len(coords)
        tot["targets"] += n_t
        tot["stranded"] += n_s
        tot["recov"] += recov
        tot["gt"] += n_gt
        print(f"{det:>7}{name:<20}{len(coords):>8}{n_t:>9}{n_s:>10}"
              f"{n_s / max(1, n_t):>10.3f}{recov:>7}{time.time() - t0:>5.0f}",
              flush=True)

    tot["det"] = det
    summary.append(tot)

print(f"\n=== pooled, against the cached det=0.99 baseline ===")
print(f"{'det':>7}{'nodes':>10}{'stranded':>10}{'str_frac':>10}"
      f"{'recoverable':>13}")
# det=0.99 baseline, from the existing cache and stranded_audit.py.
# 191512 pre-ILP nodes also cross-checks against the 191512 recorded in
# progress.md for the greedy run, so the cache really is the same pipeline.
print(f"{0.99:>7}{191512:>10}{22474:>10}{22474 / 189872:>10.3f}{278:>13}"
      f"   <- baseline, 189872 targets")
for t in summary:
    print(f"{t['det']:>7}{t['nodes']:>10}{t['stranded']:>10}"
          f"{t['stranded'] / max(1, t['targets']):>10.3f}{t['recov']:>13}")

print("\nRead the RATIO, not the levels. Stranded falling faster than nodes "
      "means the cut is landing on the useless detections. Recoverable "
      "collapsing means it is landing on the useful ones too.", flush=True)
print("DET SWEEP DONE", flush=True)
