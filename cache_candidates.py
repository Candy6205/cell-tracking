"""
cache_candidates.py -- run the GPU stage ONCE, cache the candidate edges.

Why this exists
---------------
`predict_video` is the only GPU stage. `build_graph` and `ILPSolver.solve` are
pure CPU. So every linking experiment we want to run -- edge threshold,
appearance/disappearance weights, orphan forks, component pruning, gap closing
-- can be replayed on CPU from one cached inference pass, at zero GPU cost.

What makes this possible, read verbatim from the official
`scripts/predict_unet_transformer.py`:

    def __post_init__(self) -> None:
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            ...

With `use_ilp=True` both caps stay None, and the greedy loop guards every cap
with `is not None`. So `predict_video` returns EVERY pair above `cfg.threshold`,
unconstrained, each with its probability and physical distance. Setting
`threshold` low therefore dumps the raw candidate set rather than a decision.

`sweep_edge_threshold.py`'s monotonicity argument does NOT extend to ILP: a
larger candidate set is a different optimisation, not a prefix of the old one.
That is exactly why the cache has to sit upstream of the solver.

Output, one .npz per dataset:

    coords  (N, 4) int32    t, z, y, x    -- node id is the row index
    src     (M,)   int32    source node id
    tgt     (M,)   int32    target node id
    prob    (M,)   float32  edge probability, softmax over sources
    dist    (M,)   float32  physical distance, micrometres

Consumed by `link_replay.py` on CPU.

Run it with the official CLI, which is the only path that attaches data sources
(the MCP connector cannot -- verified 2026-08-20, four field combinations):

    kaggle kernels push -p .

OUTPUT DISCIPLINE (rules 2 and 10): one line per dataset is fine at 12
datasets; nothing per-frame, nothing per-edge.
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

# ---------------------------------------------------------------- knobs

# The 12 held-out `6bba_` datasets every recorded local number comes from.
HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# Density-matched `44b6_` datasets, for the batch-vs-density diagnostic.
# DELIBERATELY EMPTY until the matching is done: `44b6` has a median 327
# cells/frame against `6bba`'s 97, and corr(edge J, density) = -0.68, so an
# unmatched `44b6` holdout scores lower for reasons that have nothing to do
# with which batch the weights saw. Pick names whose true cells/frame sit
# inside the 50-520 range the 12 above span, then fill this in.
HOLDOUT_44B6: list[str] = []

# Two datasets first. Flip to False once one run has come back clean --
# a broken 12-dataset run costs 19 minutes, a broken 2-dataset run costs 3.
SMOKE = False

CAND_THRESHOLD = 0.05   # dump everything above this; replay filters upward
TOPK = 16               # keep at most this many parents per target
DET_THRESHOLD = 0.99    # unchanged from the 0.876 submission

# ---------------------------------------------------------------- setup

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():                    # some mounts drop the "datasets/" level
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    entries = sorted(e.name for e in os.scandir("/kaggle/input")) \
        if pathlib.Path("/kaggle/input").exists() else []
    raise SystemExit(
        f"support pack not attached. /kaggle/input holds {len(entries)} "
        f"entries: {entries}. If this is empty the notebook was pushed "
        f"without data sources -- use `kaggle kernels push -p .` with "
        f"kernel-metadata.json, not the MCP connector."
    )

# Wheels that would overwrite something the Kaggle image already ships and that
# torch/scipy are compiled against. Installing these is what broke numpy.
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
    + [str(w) for w in staged],
    capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
if r.returncode != 0:
    print(r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)

import torch                                                  # noqa: E402

REPO = PACK / "repo"
for p in (REPO / "src", REPO / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))

import predict_unet_transformer as P                          # noqa: E402

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS

COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
assert COMP.exists(), "competition data not attached"
TRAIN = COMP / "train"
assert TRAIN.exists(), TRAIN

names = HOLDOUT_6BBA + HOLDOUT_44B6
if SMOKE:
    names = names[:2]
    print("[SMOKE] 2 datasets only -- set SMOKE = False for the full pass",
          flush=True)
missing = [n for n in names if not (TRAIN / f"{n}.zarr").exists()]
assert not missing, f"not in train/: {missing}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
if device.type != "cuda":
    raise SystemExit("[degraded] no GPU -- aborting rather than running "
                     "volumetric inference on CPU")

# P100 is sm_60 and the image's torch ships sm_70+ only, so every kernel
# launch dies with cudaErrorNoKernelImageForDevice. Fail here, loudly, rather
# than after the model load. Request a T4 in kernel-metadata.json.
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
if cap[0] < 7:
    raise SystemExit(
        f"[degraded] this machine is sm_{cap[0]}{cap[1]}; the image's torch "
        f"supports sm_70 and up. Set machine_shape to NvidiaTeslaT4 in "
        f"kernel-metadata.json and push again."
    )

model, window_size, downsample = P.load_model(WEIGHTS, device)
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      threshold=CAND_THRESHOLD)
print(f"threshold={cfg.threshold} det={cfg.det_threshold} "
      f"det_tta={cfg.det_tta} pool_um={cfg.pool_kernel_um} "
      f"caps={cfg.max_parents_per_node}/{cfg.max_children_per_node}",
      flush=True)
if cfg.max_parents_per_node is not None:
    print("[degraded] max_parents_per_node is set -- predict_video will "
          "greedily prune the candidate set and this cache is NOT the raw "
          "candidates it claims to be", flush=True)

# ---------------------------------------------------------------- run

OUT = pathlib.Path("/kaggle/working/cache")
OUT.mkdir(exist_ok=True)

t_all = time.time()
failed: list[str] = []

for k, name in enumerate(names, 1):
    t0 = time.time()
    try:
        coords, edges = P.predict_video(
            model, TRAIN / name, device, cfg=cfg,
            window_size=window_size, unet_batch_size=4, downsample=downsample)
    except Exception as exc:                                  # noqa: BLE001
        failed.append(name)
        print(f"[degraded] {name} FAILED ({type(exc).__name__}: {exc}) -- "
              f"no cache written for it; every replay below will silently "
              f"omit this dataset. Continuing.", flush=True)
        continue

    coords = np.asarray(coords, dtype=np.int32)
    e = np.asarray(edges, dtype=np.float64)
    if len(e) == 0:
        failed.append(name)
        print(f"[degraded] {name} produced zero candidate edges at "
              f"threshold {cfg.threshold} -- skipped", flush=True)
        continue

    src = e[:, 0].astype(np.int32)
    tgt = e[:, 1].astype(np.int32)
    prob = e[:, 2].astype(np.float32)
    dist = e[:, 3].astype(np.float32)

    # Rank candidates within each target by descending probability, keep TOPK.
    order = np.lexsort((-prob, tgt))
    tgt_sorted = tgt[order]
    first = np.r_[0, np.flatnonzero(np.diff(tgt_sorted)) + 1]
    sizes = np.diff(np.r_[first, len(tgt_sorted)])
    rank = np.arange(len(tgt_sorted)) - np.repeat(first, sizes)
    keep = order[rank < TOPK]
    n_over = int((rank >= TOPK).sum())
    if n_over:
        print(f"[degraded] {name}: {n_over} candidates dropped beyond "
              f"top-{TOPK} per target; any replay wanting more than {TOPK} "
              f"parents is truncated", flush=True)

    np.savez_compressed(OUT / f"{name}.npz", coords=coords, src=src[keep],
                        tgt=tgt[keep], prob=prob[keep], dist=dist[keep])
    mb = (OUT / f"{name}.npz").stat().st_size / 2 ** 20

    # n_above_0.5 is the submitted configuration's entire candidate set.
    # kept/n_above_0.5 is how much wider the replay can now search.
    print(f"[{k}/{len(names)}] {name}: nodes={len(coords)} "
          f"cand={len(e)} kept={len(keep)} targets={len(first)} "
          f"max_parents={int(sizes.max())} "
          f"above_0.5={int((prob > 0.5).sum())} "
          f"{mb:.1f}MB {time.time() - t0:.0f}s", flush=True)

total_mb = sum(p.stat().st_size for p in OUT.glob("*.npz")) / 2 ** 20
print(f"\n{len(list(OUT.glob('*.npz')))} cached, {total_mb:.1f} MB, "
      f"{time.time() - t_all:.0f}s", flush=True)
if failed:
    print(f"[degraded] {len(failed)} dataset(s) produced nothing: {failed}",
          flush=True)
n_cached = len(list(OUT.glob("*.npz")))
if n_cached == 0:
    raise SystemExit("[degraded] nothing was cached -- do NOT read this run "
                     "as a success")
print(f"CACHE OK ({n_cached} of {len(names)})", flush=True)
