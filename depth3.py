"""Does DEPTH cost detection? Measured from the ground-truth side.

SCRIPT_VERSION 2026-08-25a-depth3

The user watched the rendered movies and said cells at the front are
bright and cells behind are dim, then asked whether the rendering
caused it. It cannot have: the render is `vol.max(axis=0)`, a plain
maximum-intensity projection with no depth shading. So it is in the
data, and the physical reason would be attenuation -- excitation
light going in and emitted light coming out both cross tissue.

That matters because all three detector branches read brightness.

HALF THE CHAIN IS ALREADY ON RECORD: Zebrahub recall by GT-point
brightness quartile is **0.52 dimmest against 0.74 brightest**.
Dimness costs recall. The open half is whether DEPTH causes it.

A local attempt on the detection caches could not answer that: it
saw only peaks that PASSED the threshold, and attenuation removes
cells entirely. Survivors' scores actually rose slightly with z --
which is what a survivorship-limited measurement looks like, not
evidence of no effect. So this measures from the GT side, where a
missed cell is still in the denominator.

THE USER'S POSITION, recorded BEFORE the run so the result cannot be
read as more than it is: dim is dim, and there may be nothing to do
about it. This is not expected to produce a lever. It is expected to
produce a CEILING, and to stop losses being attributed to the wrong
cause -- today the same -0.5580 was attributed first to crowding and
then to tissue architecture, and depth is a third candidate nobody
had ruled out.
"""
from __future__ import annotations

import gc
import json
import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-25a-depth3"
# Assembled from nms4.py, which was assembled from nms3.py: the whole
# preamble, environment and data loading are carried unchanged so a
# difference here cannot be blamed on a changed input. Only the
# MEASUREMENT is new. Constants below that this script never reads (the
# ILP and post-link ones) are left in place for the same reason.
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
P_MIN_SWEEP = ()        # A line: measured twice, identical. Skipped.
ILP_DIV_SWEEP = (0.20, 0.50, 0.75, 1.50)   # B line, widened
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
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets, zarr_path)

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


def _frame_keys(files, prefix):
    """Frame keys are prefix + an integer, e.g. p0, p1, ... The GT arrays
    can travel in the SAME npz, and one of them is `parent_track_id`,
    which also starts with "p" -- `startswith` alone matched it and
    int("arent_track_id") killed v1 of this kernel after both sparse arms
    had already finished. The digit test is the whole fix, applied to
    every frame-key scan here rather than only the one that broke."""
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


def load_dets(name):
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in _frame_keys(z.files, "c")) + 1
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


def pair_features(dets, vels, T):
    out = []
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, 18), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        # gate query only: shrink z by GATE_Z/GATE_XY so a ball of
        # radius GATE_XY in this space IS the ellipsoid we want. The
        # distance FEATURE below stays physical and unchanged.
        _an = GATE_Z / GATE_XY
        _Sq = np.array([SCALE[0] / _an, SCALE[1], SCALE[2]])
        Pq = P * _Sq[None, :]
        Cq = C * _Sq[None, :]
        tree_q = cKDTree(Cq)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        pred = Pu + vels[t]
        pairs = []
        rows = []
        cand_of_i = defaultdict(list)
        cand_of_j = defaultdict(list)
        raw = {}
        for i, x in enumerate(Pu):
            for j in tree_q.query_ball_point(Pq[i], GATE_XY):
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
            rows.append([d, delta[0], abs(delta[0]),
                         float(np.linalg.norm(delta[1:])), md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         float(np.linalg.norm(vels[t][i])), t / max(1, T)])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out


root = find_root(None)

GATE_XY = 10.0
GATE_Z = 10.0
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
# load_image=False: the metric reads dset.tracks and dset.scale only.
# Loading the image is what demanded a GPU (io.py:199 pins it); see
# cpu-probe. Skipping it is what lets this whole line run on CPU.
        dset = open_dataset(root / "train" / name, require_tracks=True,
                              load_image=False)
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


def _bucket_by_source(coords, cand_edges):
    tcol = coords[:, 0].astype(int)
    by_t = defaultdict(list)
    for e in cand_edges:
        by_t[int(tcol[e[0]])].append(e)
    return by_t


def solve_edges(coords, cand_edges, T, div_w, window_size):
    """Solve the ILP in overlapping time windows. Returns global edge pairs.

    Window k covers frames [a, b] and k+1 covers [b, c]: they SHARE ONE
    FRAME but never share an EDGE, since k owns a->a+1 .. (b-1)->b and
    k+1 owns b->b+1 .. So the merged edge set cannot conflict and needs no
    reconciliation. Same scheme as ultrack's window_size/overlap_size.

    Measured on the 12-dataset holdout: window 25 scores 0.9382 against
    the global solve's 0.9368, at 2.08x the linking speed.
    """
    tcol = coords[:, 0].astype(int)
    by_t = _bucket_by_source(coords, cand_edges)
    if window_size is None:
        bounds = [(0, T - 1)]
    else:
        bounds, a = [], 0
        while a < T - 1:
            b = min(a + window_size, T - 1)
            bounds.append((a, b))
            a = b
    out = []
    dropped = 0
    for (a, b) in bounds:
        keep_n = np.where((tcol >= a) & (tcol <= b))[0]
        if len(keep_n) == 0:
            continue
        local_of_g = {int(g): i for i, g in enumerate(keep_n)}
        sub_coords = coords[keep_n]
        sub_edges = []
        for t in range(a, b):
            for (s, tg, p, d) in by_t.get(t, ()):
                ls, lt = local_of_g.get(int(s)), local_of_g.get(int(tg))
                if ls is not None and lt is not None:
                    sub_edges.append((ls, lt, p, d))
        if not sub_edges:
            continue
        graph = PIPE.build_graph(sub_coords, sub_edges)
        if graph.num_edges() == 0:
            continue
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=div_w)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        # Map solved nodes back BY COORDINATE, never by node_id: node_id is
        # not a stable handle across the solve (progress.md 02f7c8b).
        key_of_g = {}
        for g, c in zip(keep_n, sub_coords):
            key_of_g[(int(c[0]), int(c[1]), int(c[2]), int(c[3]))] = int(g)
        cols = {k: na[k].to_list() for k in ("node_id", "t", "z", "y", "x")}
        g_of_nid = {}
        for k in range(len(cols["node_id"])):
            key = (int(cols["t"][k]), int(round(float(cols["z"][k]))),
                   int(round(float(cols["y"][k]))),
                   int(round(float(cols["x"][k]))))
            gg = key_of_g.get(key)
            if gg is not None:
                g_of_nid[int(cols["node_id"][k])] = gg
        if graph.num_edges() == 0:
            continue
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for s_, t_ in zip(ea["source_id"].to_list(),
                          ea["target_id"].to_list()):
            gs, gt = g_of_nid.get(int(s_)), g_of_nid.get(int(t_))
            if gs is None or gt is None:
                dropped += 1
                continue
            out.append((gs, gt))
    if dropped:
        print(f"    WARN {dropped} solved edges unmatched by coordinate "
              f"-- this movie is a LOWER BOUND", flush=True)
    return out


# ---------------- the division classifier: build and train ---------------
# The unit is a TRIPLE (parent, candidate a, candidate b), not an edge.
# That is the whole point: the edge ranker scores each edge on its own and
# therefore cannot express "this long edge is a second daughter BECAUSE the
# other daughter is over there" -- the hoct paper (arXiv 2607.11754, from
# the host's own lab) makes the same argument, that edges must attend to
# one another because a division entangles two lineage paths.
#
# Features: everything, xyz included. The user's call after a long
# measurement pass, and the right one -- the in-domain validation set
# decides what survives, not a prior argument about which axis transfers.
# Both raw micrometres and spacing-normalised versions go in, so the model
# can pick whichever generalises.

MIN_SCORE_A = 0.05          # a candidate must at least be a detection
TOPK = 6                    # candidates considered per parent
LOOKAHEAD = 2               # frames a daughter must persist to count


def spacing_at(pts_um, x_um, k=5):
    if len(pts_um) <= k:
        return np.nan
    d, _ = cKDTree(pts_um).query(x_um, k=min(k + 1, len(pts_um)))
    return float(np.median(np.asarray(d)[1:]))


def triple_feats(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T):
    """One (parent i at t, candidates ja/jb at t+1) triple -> feature row."""
    p = P[i] * SCALE
    a, b = C[ja] * SCALE, C[jb] * SCALE
    va, vb = a - p, b - p
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    sep = float(np.linalg.norm(a - b))
    cosab = float(va @ vb) / max(na * nb, 1e-9)
    ang = float(np.degrees(np.arccos(np.clip(cosab, -1, 1))))
    sp = sp if np.isfinite(sp) and sp > 0 else np.nan
    far, near = max(na, nb), min(na, nb)
    vfar = va if na >= nb else vb
    return [
        na, nb, far, near, near / max(far, 1e-9),           # 0-4 distances
        sep, sep / max(sp, 1e-9), far / max(sp, 1e-9),      # 5-7 normalised
        ang, cosab,                                          # 8-9 opposition
        abs(float(va[0])), abs(float(vb[0])),                # 10-11 z parts
        abs(float(vfar[0])) / max(far, 1e-9),                # 12 z share
        float(np.linalg.norm(va[1:])),                       # 13 xy part a
        float(np.linalg.norm(vb[1:])),                       # 14 xy part b
        float(ps[i]), float(cs[ja]), float(cs[jb]),          # 15-17 scores
        min(cs[ja], cs[jb]) / max(max(cs[ja], cs[jb]), 1e-9),  # 18 sym
        float(np.linalg.norm(vel)) if vel is not None else 0.0,  # 19 speed
        sp if np.isfinite(sp) else 0.0,                      # 20 spacing
        float(len(P)), float(len(C)), float(N),              # 21-23 counts
        t / max(1, T),                                       # 24 phase
    ]


NFEAT = 25
KEEP = 0.01        # negative keep rate; positives are never dropped


NEG_PER_POS = 400        # keep every positive, thin the negatives




# ---- acceleration lookahead --------------------------------------------
# Togashi's stack item 4 (public-solutions-recon.md:313): "Acceleration
# lookahead in motion prediction (bonus for consistent acceleration), not
# just velocity." Never transplanted -- the recon lists items 1 and 5 as
# taken and leaves 2, 3, 4, 6.
#
# It is the same shape of gap div_kinematics just proved real on the
# division side: the model gets ONE temporal number. `pair_features` has
# `md`, the residual against a CONSTANT-VELOCITY extrapolation, where the
# velocity itself comes from a single previous transition. A cell that is
# turning or speeding up breaks that model, and nothing in the 18 columns
# can say so.
#
# Acceleration needs THREE positions, so it cannot exist in the two frames
# the official edge head sees, nor in a one-step velocity. That is the
# whole point.

NFEAT_BASE = 18
N_ACC = 5
NFEAT_ACC = NFEAT_BASE + N_ACC
EPS = 1e-9

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
    for k_, n_ in sorted(_WARN_COUNT.items()):
        print(f"    {k_}: {n_}", flush=True)


def pass1_tracks(dets):
    """Velocities AND the backward link map from one greedy pass.
    pass1_velocities discards the links; acceleration needs them."""
    T = len(dets)
    seeds = [seeds_of(dets[t]) for t in range(T)]
    vels = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    prev_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
            prev_of[t + 1][ci] = pi
    return seeds, vels, prev_of


def accel_of(seeds, prev_of, t, i):
    """(acceleration vector um/frame^2, steps available).

    Needs THREE positions: t, t-1, t-2. With fewer, acceleration is
    undefined -- return zero and say how many steps there were, so the
    model can split on the count instead of reading 0 as 'not
    accelerating'."""
    if t < 2:
        return np.zeros(3), 0
    i1 = int(prev_of[t][i]) if i < len(prev_of[t]) else -1
    if i1 < 0:
        warn_once("accel_no_parent",
                  "a cell had no backward link, so acceleration is "
                  "undefined -> all 5 columns are 0 and acc_hist says 0; "
                  "that is NOT a measurement of constant velocity")
        return np.zeros(3), 0
    i2 = int(prev_of[t - 1][i1]) if i1 < len(prev_of[t - 1]) else -1
    if i2 < 0:
        warn_once("accel_one_step",
                  "a cell had only ONE backward step, so acceleration "
                  "needs a third position it does not have -> 0 with "
                  "acc_hist 1")
        return np.zeros(3), 1
    P0, _ = seeds[t]
    P1, _ = seeds[t - 1]
    P2, _ = seeds[t - 2]
    if i >= len(P0) or i1 >= len(P1) or i2 >= len(P2):
        return np.zeros(3), 0
    p0 = P0[i] * SCALE
    p1 = P1[i1] * SCALE
    p2 = P2[i2] * SCALE
    return (p0 - p1) - (p1 - p2), 2


def pair_features(dets, vels, T):
    """The 18 base columns EXACTLY as the transplant computes them, then 5
    acceleration columns. `vels` is accepted and ignored -- this rebuilds
    the pass so it can keep the link map the acceleration needs."""
    seeds, vels_own, prev_of = pass1_tracks(dets)
    out = []
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, NFEAT_ACC), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        pred = Pu + vels_own[t]
        acc = np.zeros((len(P), 3))
        acc_n = np.zeros(len(P))
        for i in range(len(P)):
            a, n = accel_of(seeds, prev_of, t, i)
            acc[i] = a
            acc_n[i] = n
        # constant velocity plus half the acceleration -- the standard
        # second-order step, and what "lookahead" means here
        pred_acc = pred + 0.5 * acc
        pairs, rows = [], []
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
            md_a = float(np.linalg.norm(Cu[j] - pred_acc[i]))
            rank = next(k for k, (_, jj) in enumerate(cand_of_i[i])
                        if jj == j)
            v = vels_own[t][i]
            na = float(np.linalg.norm(acc[i]))
            nv = float(np.linalg.norm(v))
            acc_cos = (float(np.dot(v, acc[i])) / (nv * na)
                       if nv > EPS and na > EPS else 0.0)
            rows.append([d, delta[0], abs(delta[0]),
                         float(np.linalg.norm(delta[1:])), md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         nv, t / max(1, T),
                         # ---- acceleration lookahead ----
                         na, acc_cos, md_a, md - md_a, acc_n[i]])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out




from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
COLS = {"D32": list(range(NFEAT_BASE)), "D187": list(range(NFEAT_BASE))}
ARMS = ("D32", "D187")
ACC_NAMES = ["acc_len", "acc_cos", "md_acc", "md-md_acc", "acc_hist"]
DIV_W = 0.55
MATCH_UM = 5.0


# ---- the pre-NMS peaks --------------------------------------------------
PRE = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/prenms_*.npz"):
    PRE[f.stem[len("prenms_"):]] = f
print(f"\n  pre-NMS caches: {len(PRE)}", flush=True)
assert PRE, ("no prenms_*.npz found -- attach faustine649/"
             "biohub-prenms-cache. Without the peaks as they were BEFORE "
             "suppression there is nothing to compare the survivors to.")
ZPRE = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/prenmszh_*.npz"):
    ZPRE[f.stem[len("prenmszh_"):]] = f
print(f"  zebrahub pre-NMS caches: {len(ZPRE)}", flush=True)

SET_6BBA = [n for n in HOLDOUT if n in PRE and n in CACHES]
SET_44B6 = sorted(n for n in PRE
                  if n.startswith("44b6_") and n in CACHES)
SET_ZH = sorted(ZPRE)
print(f"  6bba {len(SET_6BBA)}, 44b6 {len(SET_44B6)}, "
      f"zebrahub {len(SET_ZH)}", flush=True)
assert len(SET_6BBA) >= 8, (
    f"only {len(SET_6BBA)} 6bba datasets have both caches; this arm "
    f"anchors against biohub-nms-loss's 0.9652 / 0.9561 and cannot be "
    f"compared if it is thinner")
if not SET_44B6:
    print("  WARN no 44b6 pre-NMS cache -> attach biohub-prenms2. That "
          "arm will be ABSENT, not zero.", flush=True)
if not SET_ZH:
    print("  WARN no zebrahub pre-NMS cache -> attach biohub-prenms2. "
          "**That is the only DENSE-GT arm**, so without it this run "
          "answers nothing it was built for.", flush=True)
NAMES = SET_6BBA

MATCH_UM = 5.0
BETAS = (0.30, 0.45, 0.60, 0.80)
# nms3 used (2, 3, 4, 5, 6) and every observed effect landed between
# two adjacent points. 2.0 and 3.0 were IDENTICAL -- not because 3.0
# is optimal but because nothing more is being killed below it.
FIXED = (2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5)


def _physical_nms(coords, scores, radius_um, scale=SCALE):
    """Carried VERBATIM from cache_train.py / zh_masks.py. poolsize.py
    does not define it -- it starts from caches that are already
    suppressed -- and the undefined-name pass caught its absence here.

    Winner-takes-all in PHYSICAL space: coordinates are scaled to
    micrometres first, so a radius means the same thing along z as along
    y and x despite the 4x anisotropy. It is also ORDER DEPENDENT, which
    is why a radius cannot be applied to an already-suppressed set and
    why this run reads the pre-NMS cache instead."""
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]


# ---- raw volume access -------------------------------------------------
# nms4 never touched the images -- it works entirely on cached peaks --
# so these are carried in verbatim from prenms2.py. The undefined-name
# pass caught all three; without it the script would have parsed, run for
# minutes and died inside the brightness loop.

def read_array_meta(zp):
    with open(pathlib.Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


_ZC = {}


def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(pathlib.Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(
            meta["shape"][1:])


def load_pre(name):
    z = np.load(PRE[name])
    T = max(int(k[1:]) for k in _frame_keys(z.files, "p")) + 1
    return [(np.asarray(z[f"p{t}"], float).reshape(-1, 3),
             np.asarray(z[f"q{t}"], float).reshape(-1)) for t in range(T)]


def reachable(pts_um, gt_um):
    """GT nodes with ANY point within MATCH_UM. No 1:1 constraint --
    this is an upper bound on what the linker could ever use, which is
    the right question for 'did NMS delete the cell entirely'."""
    if len(pts_um) == 0 or len(gt_um) == 0:
        return 0
    tree = cKDTree(pts_um)
    return int(sum(1 for g in gt_um
                   if len(tree.query_ball_point(g, MATCH_UM)) > 0))


def nn_median(pts_um):
    """Median nearest-neighbour distance among the peaks themselves --
    the frame's own length scale, which is what makes a dimensionless
    radius possible."""
    if len(pts_um) < 2:
        return None
    d, _ = cKDTree(pts_um).query(pts_um, k=2)
    return float(np.median(d[:, 1]))


def zh_gt_and_peaks(name):
    """Zebrahub: the peaks and the DENSE ground truth travel in the same
    npz, written by prenms2."""
    z = np.load(ZPRE[name])
    ks = _frame_keys(z.files, "p")
    assert ks, f"{name}: no p<t> peak arrays in the npz"
    T = max(int(k[1:]) for k in ks) + 1
    pre = [(np.asarray(z[f"p{t}"], float).reshape(-1, 3),
            np.asarray(z[f"q{t}"], float).reshape(-1)) for t in range(T)]
    nodes = np.asarray(z["nodes"], float) if "nodes" in z.files else None
    return pre, T, nodes


ZBINS = 8


def _zprofile(tag, names, dense):
    """Recall and GT-point brightness against depth, for one domain."""
    print(f"\n=== {tag}  ({'DENSE' if dense else 'sparse'} GT, "
          f"{len(names)} sets) ===", flush=True)
    if not names:
        print("  ABSENT", flush=True)
        return
    hit = np.zeros(ZBINS)
    tot = np.zeros(ZBINS)
    bri = np.zeros(ZBINS)
    nbri = np.zeros(ZBINS)
    detz = np.zeros(ZBINS)
    zmax_seen = 0.0

    for name in names:
        try:
            if dense:
                pre, T, gnodes = zh_gt_and_peaks(name)
                if gnodes is None:
                    continue
                g_t = np.asarray(gnodes[:, 0], int)
                g_zyx = np.asarray(gnodes[:, 1:], float)
                post = None
                vol_of = None
            else:
                gt = read_geff(geff_path(root, "train", name))
                g_t = np.asarray(gt["t"], int)
                g_zyx = np.stack([np.asarray(gt[c], float)
                                  for c in ("z", "y", "x")], axis=1)
                pre = load_pre(name)
                post = load_dets(name)
                T = min(len(pre), len(post))
                zp = zarr_path(root, "train", name)
                meta = read_array_meta(zp)
                vol_of = (zp, meta)
        except Exception as exc:                          # noqa: BLE001
            print(f"  WARN {name}: {type(exc).__name__}: "
                  f"{str(exc)[:70]} -> skipped", flush=True)
            continue

        gt_by_t = defaultdict(list)
        for r_ in range(len(g_t)):
            gt_by_t[int(g_t[r_])].append(r_)
        zmax_seen = max(zmax_seen, float(g_zyx[:, 0].max()))
        # Bin on the VOLUME's own z axis, not on the GT's range: the GT
        # may not reach the top or bottom plane, and binning on its range
        # would rescale every dataset differently and make the columns
        # incomparable.
        ZTOP = float(vol_of[1]["shape"][1]) if vol_of else 64.0
        edges = np.linspace(0.0, ZTOP, ZBINS + 1)

        for t in range(T):
            gi = gt_by_t.get(t, [])
            if not gi:
                continue
            gz = g_zyx[gi]
            gu = gz * SCALE[None, :]
            if post is not None:
                cc, ss = post[t]
                cu = np.asarray(cc, float).reshape(-1, 3)
                sv = np.asarray(ss, float).reshape(-1)
                keep = cu[sv >= UNET_THRESH]
            else:
                kc, ks = _physical_nms(pre[t][0], pre[t][1], 4.0)
                keep = kc[ks >= UNET_THRESH]
            if len(keep):
                d_, _ = cKDTree(keep * SCALE[None, :]).query(gu)
                ok = d_ <= MATCH_UM
            else:
                ok = np.zeros(len(gi), bool)
            b = np.clip(np.digitize(gz[:, 0], edges) - 1, 0, ZBINS - 1)
            for k_ in range(len(gi)):
                tot[b[k_]] += 1
                hit[b[k_]] += 1 if ok[k_] else 0
            if len(keep):
                bd = np.clip(np.digitize(keep[:, 0], edges) - 1,
                             0, ZBINS - 1)
                for v in bd:
                    detz[v] += 1

            # brightness AT the GT points, sampled from the raw volume.
            # Only every 10th annotated frame: reading the image is the
            # expensive part and the profile does not need every frame.
            if vol_of is not None and t % 10 == 0:
                try:
                    vol = load_volume(vol_of[0], t, vol_of[1])
                except Exception:
                    continue
                Z, Y, X = vol.shape
                for k_ in range(len(gi)):
                    zz = int(round(gz[k_, 0])); yy = int(round(gz[k_, 1]))
                    xx = int(round(gz[k_, 2]))
                    if not (0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X):
                        continue
                    z0, z1 = max(0, zz - 1), min(Z, zz + 2)
                    y0, y1 = max(0, yy - 3), min(Y, yy + 4)
                    x0, x1 = max(0, xx - 3), min(X, xx + 4)
                    bri[b[k_]] += float(vol[z0:z1, y0:y1, x0:x1].max())
                    nbri[b[k_]] += 1
                del vol
        gc.collect()

    print(f"  {'z plane':<12} {'GT n':>9} {'recall':>8} "
          f"{'GT bright':>11} {'det n':>9}", flush=True)
    zs, rc, bb = [], [], []
    ZTOP = 64.0
    edges = np.linspace(0.0, ZTOP, ZBINS + 1)
    for k_ in range(ZBINS):
        if tot[k_] == 0:
            print(f"  {edges[k_]:>4.0f}-{edges[k_+1]:<7.0f} "
                  f"{0:>9} {'--':>8} {'--':>11} {int(detz[k_]):>9}",
                  flush=True)
            continue
        r_ = hit[k_] / tot[k_]
        b_ = (bri[k_] / nbri[k_]) if nbri[k_] else float("nan")
        zs.append(0.5 * (edges[k_] + edges[k_ + 1]))
        rc.append(r_)
        bb.append(b_)
        print(f"  {edges[k_]:>4.0f}-{edges[k_+1]:<7.0f} "
              f"{int(tot[k_]):>9} {r_:>8.4f} "
              f"{b_:>11.1f} {int(detz[k_]):>9}", flush=True)
    print(f"  TOTAL recall {hit.sum() / max(1, tot.sum()):.4f} over "
          f"{int(tot.sum())} GT nodes", flush=True)

    if len(zs) > 3:
        zs = np.array(zs); rc = np.array(rc); bb = np.array(bb)
        print(f"  corr(z, recall)      = "
              f"{float(np.corrcoef(zs, rc)[0, 1]):+.4f}", flush=True)
        m = np.isfinite(bb)
        if m.sum() > 3:
            print(f"  corr(z, GT bright)   = "
                  f"{float(np.corrcoef(zs[m], bb[m])[0, 1]):+.4f}",
                  flush=True)
            print(f"  corr(GT bright, recall) = "
                  f"{float(np.corrcoef(bb[m], rc[m])[0, 1]):+.4f}",
                  flush=True)


print("\n=== DOES DEPTH COST DETECTION? ===", flush=True)
print("  Measured from the GT side, so a cell missed entirely is still "
      "counted -- which is exactly what a detections-only measurement "
      "cannot see.", flush=True)
print("  Already on record: Zebrahub recall by GT brightness quartile is "
      "0.52 dimmest against 0.74 brightest. Dimness costs recall. The "
      "open half is whether DEPTH causes the dimness.", flush=True)
print("  ⚠ The two end bins are contaminated: peak_local_max runs with "
      "exclude_border=False and the network's padding produces extra "
      "response at the volume boundary. Read the middle.", flush=True)

_zprofile("6bba", SET_6BBA, False)
_zprofile("44b6", SET_44B6, False)
_zprofile("zebrahub", SET_ZH, True)

print("\n=== READING IT ===", flush=True)
print("  recall AND GT brightness both falling with z closes the chain "
      "depth -> dim -> missed, and the answer is a CEILING, not a lever: "
      "the user's position, recorded before the run, is that dim is dim "
      "and there may be nothing to do about it. The value is knowing "
      "where the ceiling sits and no longer attributing losses to the "
      "wrong cause -- today's -0.5580 has already been attributed to "
      "crowding and then to tissue architecture, and depth is a third "
      "candidate nobody had ruled out.", flush=True)
print("  recall FLAT means the front/back difference on screen is not "
      "depth, and this line closes with a negative.", flush=True)
print("  Zebrahub is the load-bearing arm again: its GT is dense, so "
      "`recall` there is recall over ALL cells. On 6bba and 44b6 the "
      "denominator is the ANNOTATED cells only, and if annotation is "
      "itself z-biased the curve is about the annotators. The `det n` "
      "column is printed against `GT n` so that bias is visible.",
      flush=True)

warning_summary()
# t0 is nms4's per-domain timer and does not exist here; the
# elapsed time is not the point of this run.
print("\nDEPTH3 DONE", flush=True)
