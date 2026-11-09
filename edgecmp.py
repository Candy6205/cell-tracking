from __future__ import annotations
"""edgecmp: ref 55974245 (ours, LB 0.932) vs the public notebook fork (LB 0.935), edge by edge on the visible 4, official scorer per bucket."""
SCRIPT_VERSION = "2026-09-05a-edgecmp"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

import gc
import json
import os
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-09-04a-gap2"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
HOLDOUT = HOLDOUT_6BBA
ILP_DISAPP = 1.5
ILP_FLOOR = 0.10                            # candidates fed to the ILP
GATE_UM = 10.0
MATCH_UM = 5.0
DIV_W = 0.55

SEED_44B6 = 20260827    # dctrain.py:158 -- random, seeded, outside D32
N_44B6 = 12
N_POOL32 = 32
N_POOL175 = 175         # dctrain's pool: 199 - 12 (6bba) - 12 (44b6)

# Score the BASE primaries too (v1's arm)? Doubles stage B (~+66 min).
SCORE_BASE_ARM = False

# Recorded anchors, see the docstring. Keys: (arm, det_arm, group).
ANCHOR_HARD = {("ANCHOR_D187", "F4", "6bba"): 0.9216}
ANCHOR_SOFT = {("ANCHOR_D187", "F4", "44b6"): 0.8353,
               ("+BIDIR", "F4", "6bba"): 0.9261,     # v2, retrained here
               ("+BIDIR", "F4", "44b6"): 0.8515}
ANCHOR_DIV = {("ANCHOR_D187", "F4", "6bba"): 0.0606,     # printed, warned
              ("ANCHOR_D187", "F4", "44b6"): 0.0000,
              ("+BIDIR", "F4", "6bba"): 0.1515,
              ("+BIDIR", "F4", "44b6"): 0.0000}
if SCORE_BASE_ARM:
    ANCHOR_HARD[("ANCHOR_D187", "BASE", "6bba")] = 0.9293
    ANCHOR_SOFT[("ANCHOR_D187", "BASE", "44b6")] = 0.8185
ANCHOR_TOL = 0.0005

BUDGET_S = 10.0 * 3600  # Kaggle kills a CPU kernel at 12 h; stop cleanly first


# ---- [assembled from rankwhy.py: A preamble] ----
# chimera-style environment: pack wheels (pyscipopt for the ILP) + the
# real tracking_cellmot from a fresh clone (scoring), --no-deps so pip
# cannot half-upgrade numpy (toolchain trap).
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
for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    sys.path.insert(0, str(p))
sys.path.insert(0, "/kaggle/working/repo/src")

import numpy as np                                             # noqa: E402
from scipy.optimize import linear_sum_assignment               # noqa: E402
from scipy.spatial import cKDTree                              # noqa: E402
import tracksdata as td                                        # noqa: E402
import predict_unet_transformer as PIPE                         # noqa: E402
from tracking_cellmot.io import open_dataset                   # noqa: E402
from tracking_cellmot.metrics import (                         # noqa: E402
    evaluate as compute_metric, node_recall, per_sample_metrics, summarise)
from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "division_recovery.py").exists()), None)
assert code is not None
sys.path.insert(0, code)
import division_recovery as dr                                 # noqa: E402
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets)

# ---- [assembled from rankwhy.py: B constants] ----
SCALE = np.array([1.625, 0.40625, 0.40625])

t_all = time.time()
root = find_root(None)


def _adj_of(row):
    v = row.get("adj_edge_jaccard")
    if v is not None and v == v:
        return float(v)
    j = float(row.get("edge_jaccard", float("nan")))
    r = float(row.get("total_node_ratio", 0.0) or 0.0)
    return max(0.0, j * (1.0 - 0.1 * r))


# ---- edgecmp: the two submissions, edge by edge, on the visible 4 --------
# The user (2026-09-05): "都去做吧". Our ref 55974245 output (biohub-submit-ep
# v1, LB 0.932) against the public notebook's fork output (biohub-nb948-asis
# v1, LB 0.935), both as submission.csv, on the 4 visible movies (copies of
# train files, so both pipelines are in-sample here; the bucket-vs-bucket
# comparison is what is read, not the absolute numbers). Their nodes are
# matched to ours per frame (mutual nearest within MATCH_UM), their edges
# mapped through the match, and every bucket is scored by the official
# scorer as its own graph: precision on counted edges = tp / (tp + fp).
MATCH_UM = 5.0
VISIBLE = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
import csv                                                     # noqa: E402


def _find(cands, tail):
    for c in cands:
        p = pathlib.Path(c) / tail
        if p.exists():
            return p
    raise FileNotFoundError(f"{tail} under {cands}")


OURS_CSV = _find(["/kaggle/input/notebooks/faustine649/biohub-submit-ep",
                  "/kaggle/input/biohub-submit-ep"], "submission.csv")
THEIRS_CSV = _find(["/kaggle/input/notebooks/faustine649/biohub-nb948-asis",
                    "/kaggle/input/biohub-nb948-asis"], "submission.csv")
print(f"ours   {OURS_CSV}\ntheirs {THEIRS_CSV}", flush=True)


def load_sub(path):
    """submission.csv -> {dataset: (nodes {id: (t,z,y,x)}, edges [(s,d)])}"""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = out.setdefault(r["dataset"], ({}, []))
            if r["row_type"] == "node":
                d[0][int(r["node_id"])] = (int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
            else:
                d[1].append((int(r["source_id"]), int(r["target_id"])))
    return out


OURS, THEIRS = load_sub(OURS_CSV), load_sub(THEIRS_CSV)
print(f"ours movies {sorted(OURS)} | theirs {sorted(THEIRS)}", flush=True)
assert all(n in OURS and n in THEIRS for n in VISIBLE)


def match_nodes(a, b):
    """mutual nearest within MATCH_UM, per frame; a, b = {id: (t,z,y,x)}. -> {b_id: a_id}"""
    by_t_a, by_t_b = defaultdict(list), defaultdict(list)
    for i, (t, z, y, x) in a.items():
        by_t_a[t].append((i, z, y, x))
    for i, (t, z, y, x) in b.items():
        by_t_b[t].append((i, z, y, x))
    m = {}
    for t in by_t_b:
        if t not in by_t_a:
            continue
        ia = np.array([r[0] for r in by_t_a[t]])
        pa = np.array([r[1:] for r in by_t_a[t]], float) * SCALE
        ib = np.array([r[0] for r in by_t_b[t]])
        pb = np.array([r[1:] for r in by_t_b[t]], float) * SCALE
        ta, tb = cKDTree(pa), cKDTree(pb)
        da, ja = ta.query(pb, distance_upper_bound=MATCH_UM)      # for each b: nearest a
        db, jb = tb.query(pa, distance_upper_bound=MATCH_UM)      # for each a: nearest b
        for k in range(len(ib)):
            if np.isfinite(da[k]) and ja[k] < len(ia) and jb[ja[k]] == k:
                m[int(ib[k])] = int(ia[ja[k]])
    return m


def gt_nodes_of(name):
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    na = dset.tracks.node_attrs(attr_keys=["t", "z", "y", "x"])
    pts = np.stack([np.asarray(na[c].to_list(), float) for c in ("t", "z", "y", "x")], axis=1)
    return dset, pts


def near_gt(nodes, gt_pts):
    """fraction of nodes {id:(t,z,y,x)} with an annotated GT node within MATCH_UM in the same frame"""
    if not nodes:
        return float("nan"), 0
    by_t = defaultdict(list)
    for t, z, y, x in nodes.values():
        by_t[t].append((z, y, x))
    gt_by_t = defaultdict(list)
    for t, z, y, x in gt_pts:
        gt_by_t[int(t)].append((z, y, x))
    hit = 0
    for t, pts in by_t.items():
        if t not in gt_by_t:
            continue
        tree = cKDTree(np.array(gt_by_t[t]) * SCALE)
        d, _ = tree.query(np.array(pts) * SCALE, distance_upper_bound=MATCH_UM)
        hit += int(np.isfinite(d).sum())
    return hit / len(nodes), hit


def score_graph(dset, name, tag, nodes, edges, est):
    """nodes {id:(t,z,y,x)}, edges [(s,d)] in those ids -> official scorer row"""
    ids = list(nodes)
    pos = {i: k for k, i in enumerate(ids)}
    arr = [nodes[i] for i in ids]
    e = [(pos[s], pos[d]) for s, d in edges if s in pos and d in pos]
    gp = pathlib.Path("/kaggle/working/cmp") / f"{name}_{tag}.geff"
    gp.parent.mkdir(parents=True, exist_ok=True)
    dr._save(str(gp), arr, e)
    rr = td.graph.IndexedRXGraph.from_geff(str(gp))
    g = rr[0] if isinstance(rr, tuple) else rr
    er = compute_metric(g, dset.tracks, scale=dset.scale)
    rec = (node_recall(g, dset.tracks) if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
    row = per_sample_metrics(er, float(est) if est else float("nan"), rec)
    row["n_nodes"], row["n_edges"] = len(ids), len(e)
    return row


TAGS = ["ours", "theirs", "both", "ours_only", "theirs_only", "union"]
ROWS = defaultdict(list)
print(f"\n=== per movie: {len(VISIBLE)} visible movies, node match {MATCH_UM} um mutual ===", flush=True)
for name in VISIBLE:
    t0 = time.time()
    on, oe = OURS[name]
    tn, te = THEIRS[name]
    m = match_nodes(on, tn)                       # their id -> our id
    dset, gt_pts = gt_nodes_of(name)
    gt = read_geff(geff_path(root, "train", name))
    est = gt["extra"].get("estimated_number_of_nodes")
    our_e = set(oe)
    their_mapped = set()
    their_unmapped_edges = []
    for s, d in te:
        if s in m and d in m:
            their_mapped.add((m[s], m[d]))
        else:
            their_unmapped_edges.append((s, d))
    both = our_e & their_mapped
    ours_only = our_e - their_mapped
    theirs_only_theirids = [(s, d) for s, d in te
                            if not (s in m and d in m and (m[s], m[d]) in both)]
    # union: our nodes + their unmatched nodes (new ids), edges ours + theirs mapped
    NEW = 10_000_000
    un = dict(on)
    newid = {}
    for i, p in tn.items():
        if i not in m:
            newid[i] = NEW + i
            un[NEW + i] = p
    def uid(i):
        return m[i] if i in m else newid[i]
    union_e = set(oe) | {(uid(s), uid(d)) for s, d in te}
    unmatched_ours = {i: p for i, p in on.items() if i not in set(m.values())}
    unmatched_theirs = {i: p for i, p in tn.items() if i not in m}
    fo, ho = near_gt(unmatched_ours, gt_pts)
    ft, ht = near_gt(unmatched_theirs, gt_pts)
    print(f"  {name}: ours {len(on)} nodes / {len(oe)} edges; theirs {len(tn)} / {len(te)}; "
          f"nodes matched {len(m)} ({len(m) / len(tn):.3f} of theirs, {len(m) / len(on):.3f} of ours); "
          f"our unmatched {len(unmatched_ours)} of which {ho} ({fo:.3f}) sit on an annotated cell; "
          f"their unmatched {len(unmatched_theirs)} of which {ht} ({ft:.3f}) do", flush=True)
    print(f"    edges: both {len(both)}, ours_only {len(ours_only)}, theirs_only {len(theirs_only_theirids)} "
          f"(of which {len(their_unmapped_edges)} touch a node we do not have)", flush=True)
    graphs = {"ours": (on, oe), "theirs": (tn, te), "both": (on, sorted(both)),
              "ours_only": (on, sorted(ours_only)), "theirs_only": (tn, theirs_only_theirids),
              "union": (un, sorted(union_e))}
    for tag in TAGS:
        nn, ee = graphs[tag]
        row = score_graph(dset, name, tag, nn, ee, est)
        ROWS[tag].append(row)
        tp, fp, fn = row["edge_tp"], row["edge_fp"], row["edge_fn"]
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"    {tag:<12} nodes {row['n_nodes']:>6} edges {row['n_edges']:>6}  tp {tp:>6} fp {fp:>5} fn {fn:>5}  "
              f"precision {prec:.3f}  adj_J {_adj_of(row):.4f}  node_r {row.get('total_node_ratio', float('nan')):+.3f}",
              flush=True)
    print(f"    ({time.time() - t0:.0f}s)", flush=True)


def _adj_of(row):
    v = row.get("adj_edge_jaccard")
    if v is not None and v == v:
        return float(v)
    j = float(row.get("edge_jaccard", float("nan")))
    r = float(row.get("total_node_ratio", 0.0) or 0.0)
    return max(0.0, j * (1.0 - 0.1 * r))


print("\n=== the 4 movies together, weighted as the scorer weights (tp+fp+fn) ===", flush=True)
print(f"  {'graph':<12} {'nodes':>7} {'edges':>7} {'tp':>7} {'fp':>6} {'fn':>6} {'precision':>9} {'adj_J':>7} {'d vs ours':>9}", flush=True)
base = None
for tag in TAGS:
    rows = ROWS[tag]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(_adj_of(r) * wi for r, wi in zip(rows, w)) / sum(w)
    if tag == "ours":
        base = adj
    tp, fp, fn = (sum(r["edge_tp"] for r in rows), sum(r["edge_fp"] for r in rows),
                  sum(r["edge_fn"] for r in rows))
    print(f"  {tag:<12} {sum(r['n_nodes'] for r in rows):>7} {sum(r['n_edges'] for r in rows):>7} "
          f"{tp:>7} {fp:>6} {fn:>6} {tp / (tp + fp) if tp + fp else float('nan'):>9.3f} {adj:>7.4f} {adj - base:>+9.4f}",
          flush=True)

print("\n=== reading it ===", flush=True)
print("  Both pipelines are in-sample on these 4 movies (train copies; the official head and "
      "detector saw them, and 3 of the 4 are in our D175 ranker pool), so absolute numbers are "
      "optimistic. What is read: precision of ours_only against theirs_only, and both. ours_only "
      "precision below ~0.48 = those edges are worth dropping (intersection beats ours); "
      "theirs_only precision well above 0.48 = worth adding (union beats ours); the unmatched-"
      "node lines say whether the nodes only one side has sit on annotated cells at all.",
      flush=True)
print(f"\nEDGECMP DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("EDGECMP OK", flush=True)
