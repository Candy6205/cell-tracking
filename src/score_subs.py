"""score_subs.py -- score finished submission.csv files against train GT.

The visible test/ holds 4 copies of train movies, so any submission CSV
built from them can be scored with the official metric. This kernel scores
BOTH the standing LB 0.917 pipeline (biohub-xl-repro) and the ranker +
capacity-2 ILP candidate (biohub-submit-ranker) on the SAME 4 datasets --
a paired comparison, the only kind the family-calibration rule allows.

No detection is re-run: both CSVs come in as kernel sources. Internet is
ON here (this is not a submission kernel), so the official repo is cloned
and pip-installed the proven way.
"""
import pathlib
import subprocess
import sys

SCRIPT_VERSION = "2026-08-23q-score-subs3"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

r = subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/royerlab/"
                    "kaggle-cell-tracking-competition.git",
                    "/kaggle/working/repo"], capture_output=True, text=True)
print(f"git clone exit {r.returncode}", flush=True)

import shutil                                                  # noqa: E402
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba",
            "llvmlite", "matplotlib", "pillow", "networkx", "scikit-learn",
            "scikit-image", "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--no-deps"]
                   + [str(w) for w in sorted(WH.glob("*.whl"))],
                   capture_output=True, text=True)
print(f"pip wheels exit {r.returncode}", flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-deps", "-e", "/kaggle/working/repo"],
                   capture_output=True, text=True)
print(f"pip -e repo exit {r.returncode}", flush=True)
sys.path.insert(0, "/kaggle/working/repo/src")

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
import tracksdata as td                                        # noqa: E402
from tracking_cellmot.io import open_dataset                   # noqa: E402
from tracking_cellmot.metrics import (                         # noqa: E402
    evaluate as compute_metric, node_recall, per_sample_metrics, summarise)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "division_recovery.py").exists()), None)
assert code is not None, "code dataset not attached"
sys.path.insert(0, code)
import division_recovery as dr                                 # noqa: E402
from viz_cells import find_root, geff_path, read_geff          # noqa: E402

root = find_root(None)
print("root:", root, flush=True)

# ---- find the CSVs -----------------------------------------------------
CSVS = {}
for p in pathlib.Path("/kaggle/input").glob("*/**/submission.csv"):
    if "submit-w25" in str(p):
        tag = "w25"
    elif "submit-ranker" in str(p):
        tag = "global"
    elif "xl-repro" in str(p):
        tag = "xl0917"
    else:
        tag = str(p.parent.name)
    CSVS[tag] = p
print("CSVs found:", {k: str(v) for k, v in CSVS.items()}, flush=True)
assert len(CSVS) >= 1, "attach at least one submission kernel as source"


def score_csv(tag, csv_path):
    sub = pd.read_csv(csv_path)
    out = pathlib.Path(f"/kaggle/working/pred_{tag}")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    scored = []
    for ds, grp in sub.groupby("dataset"):
        if not (root / "train" / f"{ds}.zarr").exists():
            print(f"  {tag}/{ds}: no train GT, skipped", flush=True)
            continue
        nd = grp[grp.row_type == "node"]
        eg = grp[grp.row_type == "edge"]
        ids = nd.node_id.astype(int).to_numpy()
        rm = {int(v): i for i, v in enumerate(ids)}
        arr = np.stack([nd.t.to_numpy(float), nd.z.to_numpy(float),
                        nd.y.to_numpy(float), nd.x.to_numpy(float)], axis=1)
        pairs = []
        _bad = 0
        for a, b in zip(eg.source_id.astype(int), eg.target_id.astype(int)):
            ra, rb = rm.get(int(a)), rm.get(int(b))
            if ra is None or rb is None:
                _bad += 1
                continue
            pairs.append([ra, rb])
        if _bad:
            print(f"    WARN {tag}/{ds}: {_bad} CSV edges reference a "
                  f"node id with no node row -- dropped, this dataset "
                  f"scores as a LOWER BOUND", flush=True)
        e = np.asarray(pairs, int) if pairs else np.zeros((0, 2), int)
        dr._save(str(out / f"{ds}.geff"), arr, e)
        scored.append(ds)
    rows = []
    per = {}
    for ds in scored:
        gp = out / f"{ds}.geff"
        dset = open_dataset(root / "train" / ds, require_tracks=True)
        res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        g = res_[0] if isinstance(res_, tuple) else res_
        er = compute_metric(g, dset.tracks, scale=dset.scale)
        rec = (node_recall(g, dset.tracks)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        gt = read_geff(geff_path(root, "train", ds))
        est = gt["extra"].get("estimated_number_of_nodes")
        m = per_sample_metrics(er, float(est) if est else float("nan"), rec)
        rows.append(m)
        per[ds] = m
    return summarise(rows), per, scored


results = {}
for tag in sorted(CSVS):
    print(f"\n=== scoring {tag} ===", flush=True)
    sm, per, scored = score_csv(tag, CSVS[tag])
    results[tag] = (sm, per, scored)
    print(f"{tag}: score {sm['score']:.4f}  adj_J {sm['adj_edge_jaccard']:.4f}"
          f"  div_J {sm['division_jaccard']:.4f}  "
          f"(divTP {sm['division_tp']} FP {sm['division_fp']} "
          f"FN {sm['division_fn']})  on {len(scored)} datasets", flush=True)

print("\n=== PAIRED COMPARISON on the visible 4 ===", flush=True)
for tag, (sm, per, scored) in sorted(results.items()):
    print(f"  {tag:>8}  score {sm['score']:.4f}  adj_J "
          f"{sm['adj_edge_jaccard']:.4f}  div_J "
          f"{sm['division_jaccard']:.4f}", flush=True)
base = results.get("xl0917", (None,))[0]
if base is not None:
    for tag in sorted(results):
        if tag == "xl0917":
            continue
        d = results[tag][0]["score"] - base["score"]
        print(f"  DELTA {tag} - xl0917 = {d:+.4f}", flush=True)
if "w25" in results and "global" in results:
    d = results["w25"][0]["score"] - results["global"][0]["score"]
    print(f"  DELTA w25 - global    = {d:+.4f}", flush=True)
# per_sample_metrics carries no "score" key -- that appears only after
# summarise. That is what killed the first run, AFTER the totals printed.
for tag in sorted(results):
    per = results[tag][1]
    if not per:
        continue
    keys = next(iter(per.values())).keys()
    k = next((x for x in ("adj_edge_jaccard", "edge_jaccard") if x in keys),
             None)
    if k is None:
        print(f"  {tag}: per-sample keys = {sorted(keys)}", flush=True)
        continue
    print(f"  {tag} per-dataset {k}:", flush=True)
    for ds in sorted(per):
        print(f"    {ds}: {per[ds][k]:.4f}", flush=True)

print("\nSCORE SUBS DONE", flush=True)
