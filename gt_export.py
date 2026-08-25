"""Export the ground truth as ONE small npz per dataset.

SCRIPT_VERSION 2026-08-25a-gt-export

WHY. Every pipeline evaluation reads the GT through
`geff_path(root, "train", name)`, which points into the competition
data: **87.6 GB**, and each `.geff` is a DIRECTORY of many small chunk
files. That is fine on Kaggle, where the competition mounts for free,
and impossible anywhere else -- fetching a geff one API call at a time
is not practical, and downloading the competition is not either.

This writes the same content as `read_geff` returns, as one compact npz
per dataset, so any machine with 65 MB of detection caches and these
files can score the pipeline without the competition mount. That is what
unblocks the Colab worker (`kaggle-toolchain.md` section 11), and it is
worth having regardless: an independent copy of the GT costs a few
megabytes and removes a single point of failure.

WHICH DATASETS. The three evaluation domains that
`working-agreement.md` now requires, resolved AT RUNTIME so no dataset
name is invented here:

  * the 12 `6bba` holdout, verbatim from poolsize.py
  * 12 `44b6` outside D32's training pool, so a detection-changing sweep
    can use them without the ranker having seen them
  * (Zebrahub needs nothing -- `zhdet_*.npz` already carries its GT)

FORMAT. Keys `t`, `z`, `y`, `x`, `node_ids`, `edges`, matching what
`read_geff` yields, plus `name`. A consumer rebuilds the graph with
`division_recovery._save` and `IndexedRXGraph.from_geff`, exactly as
`zh_pipeline.py` already does for Zebrahub.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-25a-gt-export"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, (
    "viz_cells.py not found -- attach faustine649/cell-tracking. "
    "read_geff and geff_path live there and nothing else can read the "
    "competition's GT format.")
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff        # noqa: E402

root = find_root(None)
print(f"  competition root: {root}", flush=True)

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# D32's pool, so the 44b6 arm can exclude it. Read from the pkl rather
# than hardcoded: an out-of-date list here would silently export
# datasets the ranker has already seen.
import pickle                                                # noqa: E402
POOL32 = set()
for p in pathlib.Path("/kaggle/input").glob("*/**/edge_ranker_d32.pkl"):
    with open(p, "rb") as f:
        blob = pickle.load(f)
    POOL32 = set(blob.get("pool") or [])
    print(f"  D32 pool read from {p}: {len(POOL32)} datasets", flush=True)
    break
if not POOL32:
    print("  WARN edge_ranker_d32.pkl not found -> the 44b6 selection "
          "cannot exclude the ranker's training set, so those exports "
          "are NOT safe for a detection-changing sweep. Attach "
          "faustine649/biohub-poolsize.", flush=True)

TRAIN = pathlib.Path(root) / "train"
present = {p.name.replace(".zarr", "") for p in TRAIN.glob("*.zarr")}
sel_6bba = [n for n in HOLDOUT if n in present]
sel_44b6 = sorted(n for n in present
                  if n.startswith("44b6_") and n not in POOL32)[:12]
NAMES = sel_6bba + sel_44b6
print(f"  exporting {len(sel_6bba)} 6bba + {len(sel_44b6)} 44b6 = "
      f"{len(NAMES)}", flush=True)
assert len(sel_6bba) == len(HOLDOUT), (
    f"only {len(sel_6bba)} of {len(HOLDOUT)} holdout datasets are "
    f"present; an export missing some of them cannot reproduce the "
    f"recorded anchor")

OUT = pathlib.Path("/kaggle/working")
t0 = time.time()
done = 0
tot_nodes = tot_edges = 0
for name in NAMES:
    try:
        gt = read_geff(geff_path(root, "train", name))
    except Exception as e:                                   # noqa: BLE001
        print(f"  WARN {name}: read_geff failed ({str(e)[:70]}) -> this "
              f"dataset will be ABSENT and any consumer must skip it",
              flush=True)
        continue
    # estimated_number_of_nodes is the organisers' TRUE cell count for
    # the dataset, and the official metric divides by it:
    #
    #   J_adj = max(0, J * (1 - 0.1 * (N_pred - n_total) / n_total))
    #
    # v1 omitted it, so `colab_check` had nothing to pass and used the
    # ANNOTATED node count instead -- 997 where the true count is about
    # 52100 for 6bba_3abfe10a. That drove the ratio to 52.7, clamped
    # that dataset's adj_J to zero, and made Colab look like it computed
    # a different answer from Kaggle (0.1736 against 0.9254) when the
    # pipeline was in fact identical, division TP/FP/FN and all.
    #
    # Saved as a 0-d array so a consumer reads it with .item().
    _est = gt.get("extra", {}).get("estimated_number_of_nodes")
    if _est is None:
        print(f"  WARN {name}: no estimated_number_of_nodes in the geff "
              f"metadata. A consumer CANNOT compute adj_edge_jaccard for "
              f"this dataset and must skip it rather than substitute the "
              f"annotated count.", flush=True)
    out = {
        "t": np.asarray(gt["t"], np.int32),
        "z": np.asarray(gt["z"], np.float32),
        "y": np.asarray(gt["y"], np.float32),
        "x": np.asarray(gt["x"], np.float32),
        "node_ids": np.asarray(gt["node_ids"], np.int64),
        "edges": np.asarray(gt["edges"], np.int64),
        "name": np.array([name]),
        "n_total": np.array(float(_est) if _est is not None else np.nan),
    }
    np.savez_compressed(OUT / f"gt_{name}.npz", **out)
    n_n = len(out["t"])
    n_e = len(out["edges"])
    tot_nodes += n_n
    tot_edges += n_e
    done += 1
    kb = (OUT / f"gt_{name}.npz").stat().st_size / 1024
    print(f"  {done}/{len(NAMES)} {name}: {n_n} nodes, {n_e} edges, "
          f"{kb:.0f} KB", flush=True)

print(f"\n  {done} datasets, {tot_nodes} nodes, {tot_edges} edges, "
      f"{time.time() - t0:.0f}s", flush=True)
mb = sum(f.stat().st_size for f in OUT.glob("gt_*.npz")) / 2 ** 20
print(f"  total {mb:.1f} MB -- small enough to carry anywhere, which is "
      f"the point", flush=True)
print("\nGT EXPORT DONE", flush=True)
