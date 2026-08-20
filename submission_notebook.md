# Submission notebook — official pipeline + ILP

One fenced block = one Kaggle cell. Produces `/kaggle/working/submission.csv`.

Configuration this reproduces (local score **0.9269** on 12 held-out `6bba`
datasets, official `evaluate.py`):

```
--det-threshold 0.99 --use-ilp
--ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5
```

## Attach

- the competition
- `pilkwang/biohub-tracking-support-pack-50ep-v1` — weights **and** the offline
  wheels **and** a copy of the repo
- **Internet OFF** (required for submission; these cells never need it)

## Why this does not call `predict_unet_transformer.py`

`predict()` wipes its output directory on every call, so calling it per dataset
loses the previous results; and calling it once prints nothing until the end.
The loop below is ours: model loaded once, each dataset timed, CSV written
incrementally so a crash still leaves what finished.

---

## 1. Offline install

No internet. Everything comes from wheels shipped inside the support pack.

```python
import pathlib, subprocess, sys

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():                      # some mounts drop the "datasets/" level
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached"

wheels = sorted(PACK.rglob("*.whl"))
assert wheels, f"no wheels under {PACK}"
find_links = sorted({str(w.parent) for w in wheels})
print(f"{len(wheels)} wheels in {len(find_links)} dir(s)")

cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index"]
for d in find_links:
    cmd += ["--find-links", d]
cmd += [str(w) for w in wheels]
r = subprocess.run(cmd, capture_output=True, text=True)
print("pip exit", r.returncode)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
```

## 2. Imports and paths

`REPO` is the copy of the repo inside the support pack — no clone, no internet.
Both `repo/src` and `repo/scripts` go on the path, so whatever the package is
called internally resolves.

```python
import os, sys, csv, time, pathlib
import numpy as np, torch

REPO = PACK / "repo"
for p in (REPO / "src", REPO / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))

import predict_unet_transformer as P
import tracksdata as td

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS
print("config.json beside the weights:", (WEIGHTS.parent / "config.json").exists())

COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
TEST = COMP / "test"
assert TEST.exists(), TEST

names = sorted(p.stem for p in TEST.glob("*.zarr"))
assert names, f"no .zarr under {TEST}"
print(f"{len(names)} test datasets")           # 4 on commit, the real set at rescore

try:
    import pyscipopt                            # noqa: F401
    USE_ILP = True
except Exception as e:
    USE_ILP = False
    print(f"[degraded] pyscipopt unavailable ({e}) -> falling back to GREEDY "
          f"linking. That is the 0.8847 configuration, not the 0.9269 one.")
```

## 3. Inference and submission

```python
OUT     = "/kaggle/working/submission.csv"
COLUMNS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
           "source_id", "target_id")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type != "cuda":
    print("[degraded] no GPU -- this will not finish inside the time limit")

model, window_size, downsample = P.load_model(WEIGHTS, device)
cfg = P.PredictConfig(
    det_threshold=0.99,
    use_ilp=USE_ILP,
    ilp_edge_weight=-1.0,
    ilp_appearance_weight=0.0,
    ilp_disappearance_weight=1.5,
    ilp_division_weight=1.0,
)
print(f"window_size={window_size} downsample={downsample} "
      f"use_ilp={cfg.use_ilp} det_tta={cfg.det_tta}")

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
                window_size=window_size, unet_batch_size=4, downsample=downsample)
            graph = P.build_graph(coords, edges)
            if cfg.use_ilp and graph.num_edges() > 0:
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
            tt  = na["t"].to_list()
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
                  f"{time.time()-t0:.0f}s  (total {(time.time()-t_start)/60:.0f}m)",
                  flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[degraded] {name} FAILED ({type(e).__name__}: {e}) -- it "
                  f"contributes no rows, so every one of its ground-truth edges "
                  f"counts as a false negative. Continuing.", flush=True)

print(f"\n{row_id} rows, {time.time()-t_start:.0f}s total")
if failed:
    print(f"[degraded] {len(failed)} dataset(s) produced nothing: {failed}")
```

## 4. Check the file before committing

```python
import collections, csv
seen, kinds = collections.Counter(), collections.Counter()
with open(OUT) as fh:
    for row in csv.DictReader(fh):
        seen[row["dataset"]] += 1
        kinds[row["row_type"]] += 1
print("datasets in csv :", len(seen), "of", len(names), "expected")
print("row types       :", dict(kinds))
missing = sorted(set(names) - set(seen))
print("missing         :", missing if missing else "none")
print("size            :", os.path.getsize(OUT) / 2**20, "MB")
assert not missing, "a test dataset produced no rows -- fix before submitting"
```

## Before you hit Submit

- **Internet OFF**, output is exactly `/kaggle/working/submission.csv`
- no dataset name or count is hardcoded anywhere — the hidden set is swapped in
  at rescore and must be discovered from `TEST`
- the commit run only sees the **4 visible test datasets**, which are copies of
  train files, so it tells you nothing about the real runtime
- those 4 have ground truth under `train/`, so the exact file you are about to
  submit can be scored first:
  `submit_baseline.run_verify("/kaggle/working/submission.csv", gt_split="train")`
  — but they are all in the model's training data, so treat the number as a
  plumbing check, not a score
