"""ranker_scale.py -- does the edge ranker actually need absolute
micrometres?

The 18-feature edge ranker is a transplant of yusuketogashi public 0.915
solution (docs/public-solutions-recon.md, addendum 2026-08-22b). Nobody
here chose those features; the list came over wholesale, hardcoded 7 um
density radius included. Nine of the 18 are tied to an absolute length,
and corr(edge J, true density) = -0.68 with edge J running 0.968 at 50
cells/frame against 0.771 at 519.

This kernel keeps ranker_fine environment, caches, holdout and labels
EXACTLY, builds the feature rows ONCE, and trains three models that
differ only in which columns they are allowed to see:

  BASE  the 18 transplanted columns, unchanged
  EXT   those 18 plus 10 columns restating the geometry in units of the
        LOCAL cell spacing (median distance to the 5 nearest neighbours)
  PURE  scale-free only -- no absolute micrometre anywhere, no
        fixed-radius density, not even the spacing itself

Four measurements: (A) AUC under ranker_fine own protocol, comparable to
its recorded 0.9884; (B) per-holdout-dataset AUC against density; (C) the
extrapolation test -- train on the sparsest 60%, test on the densest 40%,
which is where a boosted tree on absolute thresholds should fail; (D) the
real metric, linker-only adj_J per dataset with its density correlation.

No detection re-runs. Nothing here touches a submission.
"""
from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-24a-ranker-scale"
# v4, after the daughter-stealing diagnosis (user's): the ranker's edge
# gain (+0.0055 adj_J) was cancelled by divisions because 1:1 Hungarian
# steals second daughters into wrong continuations, draining the orphan
# pool the division patch mines. Reconciliation: the ranker's LABELS
# already treat both GT children as positives, so a capacity-2 global
# solve (tracksdata ILP fed RANKER probs, division surcharge priced)
# can take both daughters natively. B line tests that; A line continues
# the Hungarian p_min sweep upward.
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
P_MIN_SWEEP = ()        # A line: measured twice, identical. Skipped.
ILP_DIV_SWEEP = (0.35, 0.45, 0.55, 0.65)   # B line, around the 0.50 peak
ILP_DISAPP = 1.5
ILP_FLOOR = 0.10                            # candidates fed to the ILP
GATE_UM = 10.0
MATCH_UM = 5.0

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
from viz_cells import find_root, geff_path, read_geff          # noqa: E402

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CAND_THR = 0.05
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
GAP1_GATE_UM = 9.0
GAP1_SNAP_UM = 3.2
GAP1_MIN_CAND_SCORE = 0.10
GAP1_CAP_FRAC = 0.003
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375

# ---- detection caches from the v2 kernel output -------------------------
BASE = pathlib.Path("/kaggle/input")
CACHES = {}
stack = [BASE]
while stack:
    d = stack.pop()
    if d.name.endswith((".zarr", ".geff")) \
            or "cell-tracking-during-development" in d.name:
        continue
    for f in d.glob("detcache_*.npz"):
        CACHES[f.stem[len("detcache_"):]] = f
    if len(d.parts) - len(BASE.parts) < 4:
        stack += [p for p in d.iterdir() if p.is_dir()]
print(f"detection caches found: {len(CACHES)}", flush=True)
assert len(CACHES) >= 40, "attach biohub-edge-ranker as kernel source"


def load_dets(name):
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]


def _logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _link_sim(prev_xyz, curr_xyz, prev_vel, prev_sc, curr_sc):
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return []
    P = prev_xyz * SCALE[None, :]
    C = curr_xyz * SCALE[None, :]
    pred = P + (0.5 * prev_vel if prev_vel is not None else 0.0)
    N, M = len(P), len(C)
    BIG = 1e9
    sim = LINK_SIM_W * np.abs(_logit(prev_sc)[:, None] - _logit(curr_sc)[None])

    def _hun(pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = np.where(Draw > gate, BIG, D) + \
            np.where(Draw > gate, 0.0, sim[np.ix_(pi, ci)])
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c]))
                for r, c in zip(ri, rc) if cost[r, c] < BIG]

    links = _hun(np.arange(N), np.arange(M), min(TIGHT_UM, MAX_LINK_UM))
    up = {p for p, _ in links}
    uc = {c for _, c in links}
    fp = np.array([i for i in range(N) if i not in up], int)
    fc = np.array([j for j in range(M) if j not in uc], int)
    return links + _hun(fp, fc, MAX_LINK_UM)


def pass1_velocities(dets):
    T = len(dets)
    vels = [np.zeros((len(seeds_of(dets[t])[0]), 3)) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
    return vels


# ---- local length scale, and the scale-invariant features ---------------
# The 18 base features are a transplant of yusuketogashi's public 0.915
# ranker (docs/public-solutions-recon.md, addendum 2026-08-22b), including
# its hardcoded 7 um density radius. Nine of the 18 are tied to an absolute
# length: d, dz, |dz|, |dxy|, md, d-md, |vel|, and the two 7 um densities.
# Cell spacing changes by an order of magnitude across this train set
# (50 -> 519 cells/frame), and corr(edge J, density) = -0.68. A boosted
# tree splits on absolute thresholds and cannot extrapolate past the
# densities it saw. These ten extra columns restate the same geometry in
# units of the LOCAL cell spacing, so the same physical situation lands on
# the same number whatever the density.

SPACING_K = 5
SPACING_FALLBACK = 7.0      # um; the transplant's own density radius
ADAPTIVE_DENS_MULT = 2.0

NFEAT_BASE = 18
EXTRA_NAMES = ["sp_p", "sp_c", "d/sp", "md/sp", "gain/sp", "xy/sp",
               "z/sp", "vel/sp", "adens_p", "adens_c"]
NFEAT_EXT = NFEAT_BASE + len(EXTRA_NAMES)

_WARN_COUNT = defaultdict(int)


def warn_once(key, msg):
    """Rule 1: a fallback that does not announce itself produces a
    plausible wrong number."""
    if _WARN_COUNT[key] == 0:
        print(f"  WARN {msg}", flush=True)
    _WARN_COUNT[key] += 1


def warning_summary():
    if not _WARN_COUNT:
        print("  no degradations fired", flush=True)
        return
    print("  degradation summary:", flush=True)
    for k, n in sorted(_WARN_COUNT.items()):
        print(f"    {k}: {n}", flush=True)


def local_spacing(pts_um, k=SPACING_K):
    """Median distance to the k nearest neighbours, per point, in um.

    This is the length scale every ratio feature divides by."""
    n = len(pts_um)
    if n == 0:
        return np.zeros(0)
    if n <= k:
        warn_once("spacing_tiny_frame",
                  f"a frame holds {n} cells <= k={k}, so it has no local "
                  f"spacing -> substituting {SPACING_FALLBACK} um; every "
                  f"ratio feature in such frames is that constant, not a "
                  f"measurement")
        return np.full(n, SPACING_FALLBACK)
    d, _ = cKDTree(pts_um).query(pts_um, k=k + 1)
    sp = np.median(np.asarray(d)[:, 1:], axis=1)
    bad = ~np.isfinite(sp) | (sp <= 0)
    if bad.any():
        good = sp[~bad]
        sub = float(np.median(good)) if len(good) else SPACING_FALLBACK
        warn_once("spacing_degenerate",
                  f"{int(bad.sum())} points had non-positive spacing "
                  f"(coincident detections) -> substituting {sub:.2f} um; "
                  f"their ratio features are that approximation")
        sp = np.where(bad, sub, sp)
    return sp


def pair_features(dets, vels, T):
    """Base 18 columns EXACTLY as the transplant computed them, then 10
    scale-invariant columns. One build, two column views -- the A/B can
    differ only in which columns the model sees."""
    out = []
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, NFEAT_EXT), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        sp_p = local_spacing(Pu)
        sp_c = local_spacing(Cu)
        adens_p = np.array([len(tree_p.query_ball_point(
            x, ADAPTIVE_DENS_MULT * sp_p[i])) - 1 for i, x in enumerate(Pu)])
        adens_c = np.array([len(tree_c.query_ball_point(
            x, ADAPTIVE_DENS_MULT * sp_c[j])) - 1 for j, x in enumerate(Cu)])
        pred = Pu + vels[t]
        pairs = []
        rows = []
        cand_of_i = defaultdict(list)
        cand_of_j = defaultdict(list)
        raw = {}
        for i, x in enumerate(Pu):
            for j in tree_c.query_ball_point(x, GATE_UM):
                d = float(np.linalg.norm(Cu[j] - x))
                raw[(i, j)] = d
                cand_of_i[i].append((d, j))
                cand_of_j[j].append((d, i))
        for i in cand_of_i:
            cand_of_i[i].sort()
        for (i, j), d in raw.items():
            delta = Cu[j] - Pu[i]
            md = float(np.linalg.norm(Cu[j] - pred[i]))
            rank = next(k for k, (_, jj) in enumerate(cand_of_i[i])
                        if jj == j)
            xy = float(np.linalg.norm(delta[1:]))
            dz = abs(float(delta[0]))
            vmag = float(np.linalg.norm(vels[t][i]))
            s = float(sp_p[i])
            rows.append([d, delta[0], dz, xy, md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         vmag, t / max(1, T),
                         # ---- scale-invariant block ----
                         s, float(sp_c[j]),
                         d / s, md / s, (d - md) / s, xy / s, dz / s,
                         vmag / s,
                         adens_p[i], adens_c[j]])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out


root = find_root(None)


# ---- graph plumbing (xl pipeline tail) ---------------------------------
def _dist_um(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) * SCALE
    return float(np.sqrt((d * d).sum()))


def _short_filter(nodes, edges):
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out_count = defaultdict(int)
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        out_count[a] += 1
    comps = defaultdict(list)
    for nid in nodes:
        comps[find(nid)].append(nid)
    keep = set()
    for members in comps.values():
        if len(members) >= SHORT_MIN or \
                any(out_count[n] >= 2 for n in members):
            keep.update(members)
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}
    edges2 = [(a, b) for a, b in edges if a in nodes2 and b in nodes2]
    return nodes2, edges2


def _linefit(nodes, edges):
    pred = defaultdict(list)
    succ = defaultdict(list)
    for a, b in edges:
        if int(nodes[b]["t"]) == int(nodes[a]["t"]) + 1:
            succ[a].append(b)
            pred[b].append(a)
    orig = {k: v["xyz"].copy() for k, v in nodes.items()}
    updates = {}
    for nid in nodes:
        neigh = [(0, nid)]
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ps = pred.get(cur, [])
            if len(ps) != 1:
                break
            cur = ps[0]
            neigh.append((-step, cur))
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ss = succ.get(cur, [])
            if len(ss) != 1:
                break
            cur = ss[0]
            neigh.append((step, cur))
        if len(neigh) < 3:
            continue
        dt = np.array([a for a, _ in neigh], float)
        xyz = np.stack([orig[n] for _, n in neigh])
        fit = np.array([np.polyval(np.polyfit(dt, xyz[:, ax], 1), 0.0)
                        for ax in range(3)])
        if np.isfinite(fit).all():
            updates[nid] = (1 - LINEFIT_WEIGHT) * orig[nid] + \
                LINEFIT_WEIGHT * fit
    for nid, xyz in updates.items():
        nodes[nid]["xyz"] = xyz


def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    T = len(cand)
    succ = {}
    pred = {}
    for a, b in edges:
        succ[a] = b
        pred[b] = a
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    cand_used = [np.zeros(len(c), bool) for c in cand]
    max_gaps = max(1, int(GAP1_CAP_FRAC * len(nodes)))
    n_added = 0
    BIG = 1e9
    for t in range(T - 2):
        if n_added >= max_gaps:
            break
        ends = [g for g in by_t.get(t, []) if g not in succ and g in pred]
        starts = [g for g in by_t.get(t + 2, [])
                  if g not in pred and g in succ]
        if not ends or not starts:
            continue
        P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
        C = np.asarray([nodes[g]["xyz"] for g in starts], float) * SCALE
        D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2))
        cost = np.where(D > GAP1_GATE_UM, BIG, D)
        ri, ci = linear_sum_assignment(cost)
        for r_, c_ in zip(ri, ci):
            if n_added >= max_gaps or cost[r_, c_] >= BIG:
                continue
            ge, gs = ends[r_], starts[c_]
            if ge in succ or gs in pred:
                continue
            mid = 0.5 * (nodes[ge]["xyz"] + nodes[gs]["xyz"])
            tree = cand_trees[t + 1]
            if tree is None:
                continue
            dist, idx = tree.query(mid * SCALE)
            if dist > GAP1_SNAP_UM or cand_used[t + 1][idx] or \
                    cand_sc[t + 1][idx] < GAP1_MIN_CAND_SCORE:
                continue
            use = np.asarray(cand[t + 1][idx], float)
            cand_used[t + 1][idx] = True
            ng = next_id
            next_id += 1
            nodes[ng] = {"t": t + 1, "xyz": use}
            edges.append((ge, ng))
            edges.append((ng, gs))
            succ[ge] = ng
            pred[ng] = ge
            succ[ng] = gs
            pred[gs] = ng
            n_added += 1
    return next_id


def _add_safe_divisions(nodes, edges):
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    orphan_tree = {}
    orphan_ids = {}
    for t, ids in by_t.items():
        orph = [g for g in ids if not pred.get(g)]
        orphan_ids[t] = orph
        if orph:
            orphan_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in orph], float) * SCALE)
    proposals = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = orphan_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = orphan_ids[t + 1][idx]
            if q == c1:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            d_q, q_idx = tree.query(nodes[c1]["xyz"] * SCALE)
            if orphan_ids[t + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            if _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s \
                    < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue
        edges.append((p, q))
        succ[p].append(q)
        pred[q].append(p)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    return added


def score_dir(pred_dir):
    rows = []
    for gp in sorted(pathlib.Path(pred_dir).glob("*.geff")):
        name = gp.stem
        dset = open_dataset(root / "train" / name, require_tracks=True)
        res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        g = res_[0] if isinstance(res_, tuple) else res_
        er = compute_metric(g, dset.tracks, scale=dset.scale)
        rec = (node_recall(g, dset.tracks)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        gt = read_geff(geff_path(root, "train", name))
        est = gt["extra"].get("estimated_number_of_nodes")
        rows.append(per_sample_metrics(er, float(est) if est else
                                       float("nan"), rec))
    return summarise(rows)




# ---- build features + labels once, keep them PER DATASET ----------------
# Per-dataset arrays are what make the density-stratified splits possible.
ALL_NAMES = sorted(CACHES)
train_names = [n for n in ALL_NAMES if n not in HOLDOUT]
print(f"\n=== FEATURES: {len(train_names)} train + "
      f"{len(HOLDOUT)} holdout datasets ===", flush=True)

DS = {}          # name -> dict(X, y, dets, feats, T, cells, spacing)
t0 = time.time()
for k_ds, name in enumerate(ALL_NAMES):
    dets = load_dets(name)
    T = len(dets)
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)

    # density descriptors of this dataset
    per_frame_cells = [len(seeds_of(dets[t])[0]) for t in range(T)]
    sps = []
    for t in range(0, T, max(1, T // 10)):
        Pu = seeds_of(dets[t])[0] * SCALE[None, :]
        if len(Pu) > SPACING_K:
            sps.append(float(np.median(local_spacing(Pu))))
    cells = float(np.median(per_frame_cells)) if per_frame_cells else 0.0
    spacing = float(np.median(sps)) if sps else float("nan")

    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gt_edges = set()
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            gt_edges.add((ra, rb))
    child_of = {}
    for (a, b) in gt_edges:
        child_of.setdefault(a, []).append(b)
    gt_by_t = defaultdict(list)
    for i in range(len(g_t)):
        gt_by_t[g_t[i]].append(i)

    Xl, yl = [], []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        if not pairs:
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]

        def match(pos_u, gt_idx_list):
            if not gt_idx_list:
                return {}
            gtree = cKDTree(g_pos[gt_idx_list])
            m = {}
            for di, x in enumerate(pos_u):
                dd, gi = gtree.query(x)
                if dd <= MATCH_UM:
                    m[di] = gt_idx_list[int(gi)]
            return m

        m_p = match(Pu, gt_by_t.get(t, []))
        m_c = match(Cu, gt_by_t.get(t + 1, []))
        det_of_gt = {g: d for d, g in m_c.items()}
        for (i, j), row in zip(pairs, rows):
            if i not in m_p:
                continue
            kids = child_of.get(m_p[i], [])
            if not kids:
                continue
            j_true = {det_of_gt.get(b) for b in kids}
            j_true.discard(None)
            Xl.append(row)
            yl.append(1 if j in j_true else 0)

    DS[name] = {
        "X": np.asarray(Xl, np.float32) if Xl else np.zeros((0, NFEAT_EXT),
                                                            np.float32),
        "y": np.asarray(yl, np.int8),
        "dets": dets, "feats": feats, "T": T,
        "cells": cells, "spacing": spacing,
    }
    if (k_ds + 1) % 10 == 0 or k_ds == len(ALL_NAMES) - 1:
        print(f"  {k_ds + 1}/{len(ALL_NAMES)} datasets "
              f"({time.time() - t0:.0f}s)", flush=True)

warning_summary()

dens_of = {n: DS[n]["cells"] for n in ALL_NAMES}
sp_of = {n: DS[n]["spacing"] for n in ALL_NAMES}
print(f"\n  density span: {min(dens_of.values()):.0f} -> "
      f"{max(dens_of.values()):.0f} cells/frame", flush=True)
print(f"  spacing span: {np.nanmin(list(sp_of.values())):.2f} -> "
      f"{np.nanmax(list(sp_of.values())):.2f} um", flush=True)


# ---- the three column views --------------------------------------------
COLS = {
    # the transplant, unchanged
    "BASE": list(range(NFEAT_BASE)),
    # transplant + the scale-invariant block
    "EXT": list(range(NFEAT_EXT)),
    # scale-free only: no absolute um anywhere, no fixed-radius density,
    # not even the spacing itself -- the strongest form of the hypothesis
    "PURE": [6, 7, 8, 11, 12, 13, 14, 15, 17] + list(range(20, NFEAT_EXT)),
}
ARMS = ("BASE", "EXT", "PURE")


def fit(names, arm, seed=0):
    X = np.concatenate([DS[n]["X"][:, COLS[arm]] for n in names])
    y = np.concatenate([DS[n]["y"] for n in names])
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    m.fit(X, y)
    return m


def auc_on(model, names, arm):
    X = np.concatenate([DS[n]["X"][:, COLS[arm]] for n in names])
    y = np.concatenate([DS[n]["y"] for n in names])
    if y.sum() == 0 or y.sum() == len(y):
        warn_once("auc_degenerate",
                  f"a split has {int(y.sum())} positives out of {len(y)} -> "
                  f"AUC undefined, reporting nan instead of a number")
        return float("nan"), len(y)
    return roc_auc_score(y, model.predict_proba(X)[:, 1]), len(y)


# ================= STAGE A: overall AUC, same protocol ==================
print("\n=== STAGE A: AUC on a random 10% split of the training pool ===",
      flush=True)
print("  (ranker_fine recorded 0.9884 for BASE under this protocol)",
      flush=True)
models = {}
for arm in ARMS:
    X = np.concatenate([DS[n]["X"][:, COLS[arm]] for n in train_names])
    y = np.concatenate([DS[n]["y"] for n in train_names])
    perm = np.random.default_rng(0).permutation(len(y))
    n_val = max(1000, len(y) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(X[ti], y[ti])
    a = roc_auc_score(y[vi], m.predict_proba(X[vi])[:, 1])
    print(f"  {arm:<5} {len(COLS[arm]):>2} cols   AUC {a:.4f}   "
          f"rows {len(y)}, {y.mean():.3f} pos", flush=True)
    models[arm] = fit(train_names, arm)      # refit on the whole pool
    with open(f"/kaggle/working/edge_ranker_{arm.lower()}.pkl", "wb") as fh:
        pickle.dump({"model": models[arm], "cols": COLS[arm],
                     "version": SCRIPT_VERSION}, fh)


# ============ STAGE B: does AUC still slide with density? ===============
print("\n=== STAGE B: per-holdout-dataset AUC vs density ===", flush=True)
hold_sorted = sorted(HOLDOUT, key=lambda n: dens_of[n])
head = "  dataset            cells/fr  spacing"
for arm in ARMS:
    head += f"   {arm:>6}"
print(head, flush=True)
per_ds = {a: [] for a in ARMS}
for n in hold_sorted:
    line = f"  {n:<18} {dens_of[n]:>8.0f}  {sp_of[n]:>7.2f}"
    for arm in ARMS:
        a, _ = auc_on(models[arm], [n], arm)
        per_ds[arm].append(a)
        line += f"   {a:.4f}"
    print(line, flush=True)

dv = np.array([dens_of[n] for n in hold_sorted], float)
print("\n  corr(AUC, density) and sparse-vs-dense gap:", flush=True)
for arm in ARMS:
    v = np.array(per_ds[arm], float)
    ok = np.isfinite(v)
    c = float(np.corrcoef(dv[ok], v[ok])[0, 1]) if ok.sum() > 2 \
        else float("nan")
    half = max(1, int(ok.sum()) // 2)
    gap = float(np.nanmean(v[ok][:half]) - np.nanmean(v[ok][half:]))
    print(f"  {arm:<5} corr {c:+.3f}   sparse-half minus dense-half "
          f"{gap:+.4f}", flush=True)


# ====== STAGE C: the extrapolation test -- train sparse, test dense =====
print("\n=== STAGE C: train on the SPARSEST 60%, test on the DENSEST 40% "
      "(training pool only) ===", flush=True)
print("  A boosted tree splits on absolute thresholds. If that is the "
      "problem, the dense half sits past every split BASE learned.",
      flush=True)
by_dens = sorted(train_names, key=lambda n: dens_of[n])
cut = int(len(by_dens) * 0.6)
sparse, dense = by_dens[:cut], by_dens[cut:]
print(f"  sparse train: {len(sparse)} datasets, "
      f"{dens_of[sparse[0]]:.0f}-{dens_of[sparse[-1]]:.0f} cells/frame",
      flush=True)
print(f"  dense test:   {len(dense)} datasets, "
      f"{dens_of[dense[0]]:.0f}-{dens_of[dense[-1]]:.0f} cells/frame",
      flush=True)
for arm in ARMS:
    m = fit(sparse, arm)
    a_out, nrow = auc_on(m, dense, arm)
    a_in, _ = auc_on(m, sparse, arm)
    print(f"  {arm:<5} in-domain {a_in:.4f}   EXTRAPOLATED {a_out:.4f}   "
          f"drop {a_in - a_out:+.4f}   ({nrow} test rows)", flush=True)

print("  control, reverse direction (train dense -> test sparse):",
      flush=True)
for arm in ARMS:
    m = fit(dense, arm)
    a_out, _ = auc_on(m, sparse, arm)
    print(f"  {arm:<5} {a_out:.4f}", flush=True)


# ====== STAGE D: the real metric -- linker-only adj_J per dataset ========
# Hungarian at P_MIN, then short filter + linefit. No gap closing and no
# safe divisions: fewer moving parts, and both arms get the identical tail,
# so the COMPARISON is valid even though the absolute number sits below
# the recorded full-pipeline 0.9318.
P_MIN = 0.20
print(f"\n=== STAGE D: linker-only adj_J on holdout, p_min={P_MIN} ===",
      flush=True)
print("  linker + short filter + linefit; no gap closing, no safe "
      "divisions. Both arms take the identical tail.", flush=True)


def link_and_score(arm):
    model = models[arm]
    PRED = pathlib.Path(f"/kaggle/working/link_{arm.lower()}")
    PRED.mkdir(exist_ok=True)
    for name in HOLDOUT:
        d = DS[name]
        dets, feats, T = d["dets"], d["feats"], d["T"]
        nodes = {}
        frame_ids = []
        nid = 1
        for t in range(T):
            c_all, s_all = np.asarray(dets[t][0], float), \
                np.asarray(dets[t][1], float)
            ids = []
            for xyz in c_all[s_all >= UNET_THRESH]:
                nodes[nid] = {"t": t, "xyz": np.asarray(xyz, float)}
                ids.append(nid)
                nid += 1
            frame_ids.append(ids)
        edges = []
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            if not pairs:
                continue
            pr = model.predict_proba(rows[:, COLS[arm]])[:, 1]
            N, M = len(P), len(C)
            BIG = 1e9
            cost = np.full((N, M), BIG)
            for (i, j), p in zip(pairs, pr):
                if p >= P_MIN:
                    cost[i, j] = 1.0 - p
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] < BIG:
                    edges.append((frame_ids[t][int(a)],
                                  frame_ids[t + 1][int(b)]))
        nodes2, edges2 = _short_filter(nodes, edges)
        _linefit(nodes2, edges2)
        ids = sorted(nodes2)
        remap = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes2[g]["t"], *nodes2[g]["xyz"]] for g in ids],
                       float)
        e2 = np.array([[remap[a], remap[b]] for a, b in edges2], int)
        dr._save(str(PRED / f"{name}.geff"), arr, e2)
    out = {}
    for gp in sorted(PRED.glob("*.geff")):
        nm = gp.stem
        dset = open_dataset(root / "train" / nm, require_tracks=True)
        res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        g = res_[0] if isinstance(res_, tuple) else res_
        er = compute_metric(g, dset.tracks, scale=dset.scale)
        rec = (node_recall(g, dset.tracks)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        gt = read_geff(geff_path(root, "train", nm))
        est = gt["extra"].get("estimated_number_of_nodes")
        out[nm] = summarise([per_sample_metrics(
            er, float(est) if est else float("nan"), rec)])
    return out


res = {}
for arm in ARMS:
    ta = time.time()
    res[arm] = link_and_score(arm)
    print(f"  {arm} linked and scored ({time.time() - ta:.0f}s)", flush=True)

head = "  dataset            cells/fr"
for arm in ARMS:
    head += f"   {arm:>6}"
print("\n  per-dataset adj_edge_jaccard:", flush=True)
print(head, flush=True)
adj = {a: [] for a in ARMS}
for n in hold_sorted:
    line = f"  {n:<18} {dens_of[n]:>8.0f}"
    for arm in ARMS:
        v = float(res[arm][n]["adj_edge_jaccard"])
        adj[arm].append(v)
        line += f"   {v:.4f}"
    print(line, flush=True)

print("\n  MEAN adj_J, corr with density, and the composite score:",
      flush=True)
for arm in ARMS:
    v = np.array(adj[arm], float)
    ok = np.isfinite(v)
    c = float(np.corrcoef(dv[ok], v[ok])[0, 1]) if ok.sum() > 2 \
        else float("nan")
    sc = float(np.nanmean([res[arm][n]["score"] for n in hold_sorted]))
    print(f"  {arm:<5} adj_J {np.nanmean(v):.4f}   "
          f"corr(adj_J, density) {c:+.3f}   score {sc:.4f}", flush=True)

print("\n  Reference: corr(edge J, true density) = -0.68 was measured on "
      "the full pipeline (progress.md). The number above is linker-only, "
      "so compare the ARMS to each other, not to -0.68.", flush=True)

warning_summary()
print("\nRANKER SCALE DONE", flush=True)
