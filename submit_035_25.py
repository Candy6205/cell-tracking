"""
submit_035_25.py -- the 0.876 submission with two numbers changed.

    threshold             0.50 -> 0.35
    ilp_disappearance     1.5  -> 2.5

Everything else is byte-for-byte the configuration that scored 0.876 on
2026-08-19: same weights, det_threshold 0.99, appearance 0.0, division 1.0.

Why those two
-------------
`threshold` defaults to 0.5 and the 0.876 run never set it. With `use_ilp=True`
the greedy caps stay None, so it is the ONLY limit on what the linker may
consider. The edge head softmaxes over sources, so a target whose parents split
the mass below 0.5 emits no candidate at all: 11.8% of targets, and 278 of the
450 false negatives, measured.

Opening the gate alone over-links, which is what `disappearance` corrects --
raising it makes the solver prefer long continuous tracks and discard scattered
nodes. Swept on the cached candidates over 12 held-out datasets:

    thr   disapp   adj_J
    0.50  1.5      0.9269   <- the 0.876 submission, reproduced exactly
    0.35  1.5      0.9296
    0.35  2.5      0.9337   <- this run, +0.0068
    0.25  2.5      0.9302

Local is optimistic by 0.051 and only the PAIRED delta transfers. The point of
submitting is as much the second calibration point as the +0.0068: the offset
currently rests on a single observation.

Runtime: the 0.876 run measured 94.2 s/dataset including install, ~5.2 h for
~199 volumes against a 12 h limit. A lower gate enlarges the ILP but inference
dominates; the sweep put the solver at roughly 1.5x, so ~4-6 h is expected.

Machine: the image's torch is sm_70+. A P100 is sm_60 and every kernel launch
dies. kernel-metadata.json must pin NvidiaTeslaT4; this asserts it rather than
discovering it 200 datasets in.
"""

from __future__ import annotations

import csv
import os
import pathlib
import shutil
import subprocess
import sys
import time

SCRIPT_VERSION = "submit-2026-08-20a"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

THRESHOLD = 0.35
DISAPPEARANCE = 2.5
DET_THRESHOLD = 0.99
APPEARANCE = 0.0
DIVISION = 1.0

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached"

# Wheels that would overwrite something the Kaggle image already ships and that
# torch/scipy are compiled against. Installing these is what broke numpy.
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}

WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
excluded = []
for w in sorted(PACK.rglob("*.whl")):
    dist = w.name.split("-")[0].replace("_", "-").lower()
    if dist in KEEP_OUT:
        excluded.append(w.name)
    else:
        shutil.copy(w, WH / w.name)
staged = sorted(WH.glob("*.whl"))
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    + [str(w) for w in staged], capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels, {len(excluded)} held "
      f"back)", flush=True)
if r.returncode != 0:
    print(r.stdout[-1500:], flush=True)
    print(r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

REPO = PACK / "repo"
for p in (REPO / "src", REPO / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))

import predict_unet_transformer as P                          # noqa: E402
import tracksdata as td                                       # noqa: E402

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS

COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
TEST = COMP / "test"
assert TEST.exists(), TEST

names = sorted(p.stem for p in TEST.glob("*.zarr"))
assert names, f"no .zarr under {TEST}"
print(f"{len(names)} test datasets", flush=True)   # 4 on commit, ~199 at rescore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
assert device.type == "cuda", "no GPU -- this cannot finish inside the limit"
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
assert cap[0] >= 7, (
    f"sm_{cap[0]}{cap[1]}: the image's torch supports sm_70 and up, so every "
    f"kernel launch would fail. Pin machine_shape to NvidiaTeslaT4."
)

try:
    import pyscipopt                                          # noqa: F401
except Exception as exc:                                      # noqa: BLE001
    raise SystemExit(
        f"[degraded] pyscipopt unavailable ({exc}). Without it the ILP cannot "
        f"run and the fallback is greedy linking -- a different configuration "
        f"scoring ~0.88 locally, not the one this notebook claims to submit. "
        f"Aborting rather than submitting something mislabelled."
    )

model, window_size, downsample = P.load_model(WEIGHTS, device)
cfg = P.PredictConfig(
    det_threshold=DET_THRESHOLD,
    use_ilp=True,
    threshold=THRESHOLD,
    ilp_edge_weight=-1.0,
    ilp_appearance_weight=APPEARANCE,
    ilp_disappearance_weight=DISAPPEARANCE,
    ilp_division_weight=DIVISION,
)
print(f"threshold={cfg.threshold} det={cfg.det_threshold} "
      f"det_tta={cfg.det_tta} pool_um={cfg.pool_kernel_um} "
      f"caps={cfg.max_parents_per_node}/{cfg.max_children_per_node} "
      f"app={APPEARANCE} disapp={DISAPPEARANCE} div={DIVISION}", flush=True)

OUT = "/kaggle/working/submission.csv"
COLUMNS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
           "source_id", "target_id")

t_start = time.time()
row_id, failed = 0, []

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(COLUMNS)

    for k, name in enumerate(names, 1):
        t0 = time.time()
        try:
            coords, edges = P.predict_video(
                model, TEST / name, device, cfg=cfg,
                window_size=window_size, unet_batch_size=4,
                downsample=downsample)
            graph = P.build_graph(coords, edges)
            if graph.num_edges() > 0:
                solver = td.solvers.ILPSolver(
                    edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                    appearance_weight=cfg.ilp_appearance_weight,
                    disappearance_weight=cfg.ilp_disappearance_weight,
                    division_weight=cfg.ilp_division_weight,
                )
                with P.suppress_output():
                    graph = solver.solve(graph)

            na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
            nid = na["node_id"].to_list()
            tt = na["t"].to_list()
            zz, yy, xx = (na[c].to_list() for c in ("z", "y", "x"))
            for j in range(len(nid)):
                w.writerow([row_id, name, "node", int(nid[j]), int(tt[j]),
                            int(round(float(zz[j]))), int(round(float(yy[j]))),
                            int(round(float(xx[j]))), -1, -1])
                row_id += 1

            n_e = 0
            if graph.num_edges() > 0:
                ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
                src, tgt = ea["source_id"].to_list(), ea["target_id"].to_list()
                n_e = len(src)
                for j in range(n_e):
                    w.writerow([row_id, name, "edge", -1, -1, -1, -1, -1,
                                int(src[j]), int(tgt[j])])
                    row_id += 1
            fh.flush()
            print(f"[{k}/{len(names)}] {name}: {len(nid)} nodes, {n_e} edges, "
                  f"{time.time() - t0:.0f}s  "
                  f"(total {(time.time() - t_start) / 60:.0f}m)", flush=True)
        except Exception as e:                                # noqa: BLE001
            failed.append(name)
            print(f"[degraded] {name} FAILED ({type(e).__name__}: {e}) -- it "
                  f"contributes no rows, so every one of its ground-truth "
                  f"edges counts as a false negative. Continuing.", flush=True)

print(f"\n{row_id} rows, {time.time() - t_start:.0f}s total", flush=True)
if failed:
    print(f"[degraded] {len(failed)} dataset(s) produced nothing: {failed}",
          flush=True)

import collections                                            # noqa: E402

seen, kinds = collections.Counter(), collections.Counter()
with open(OUT) as fh:
    for row in csv.DictReader(fh):
        seen[row["dataset"]] += 1
        kinds[row["row_type"]] += 1
print("datasets in csv :", len(seen), "of", len(names), "expected")
print("row types       :", dict(kinds))
missing = sorted(set(names) - set(seen))
print("missing         :", missing if missing else "none")
print("size            :", os.path.getsize(OUT) / 2 ** 20, "MB")
assert not missing, "a test dataset produced no rows -- do not submit this"
print("SUBMISSION READY", flush=True)
