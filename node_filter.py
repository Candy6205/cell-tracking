"""A node filter: drop hallucinated detections before the linker.

SCRIPT_VERSION 2026-08-25a-node-filter

WHERE THIS COMES FROM. `biohub-node-bright` measured, on Zebrahub's dense
GT with greedy 1:1 matching and NO masks, that box brightness separates a
real detection from a hallucinated one far better than the detector's own
score does:

    SCORE only   AUC 0.7077
    BRIGHT only  AUC 0.8343
    BOTH         AUC 0.8495      -> +0.1418 over the score alone

and, stratified by what the false positive actually IS:

    vs EMPTY     (nothing there)   SCORE 0.7034  BRIGHT 0.8549  BOTH 0.8659
    vs DUPLICATE (NMS failure)     SCORE 0.7700  BRIGHT 0.6683  BOTH 0.7555

Brightness cannot separate a DUPLICATE from a real cell -- a duplicate
sits ON a real cell and is exactly as bright, and the medians confirm it
(DUP c_mean 0.3363 against REAL 0.3346, very slightly HIGHER). That would
have killed the line, except the false positives are not duplicates:
**EMPTY 7,566 against DUPLICATE 496 on the validation split, so 94% of
them are hallucinations** -- precisely what brightness can see.

WHAT THIS SCRIPT DOES. Train that filter on Zebrahub, then drop the
lowest-scoring detections on the competition holdout BEFORE anything
else runs, and score adj_J. Filtering at load time means the ranker, the
ILP and the whole post-processing tail see the reduced set with no
further wiring.

THE CONTROL, and without it the run is worthless. Dropping detections
raises precision by construction. The question is whether the FILTER
beats simply raising the detection threshold to drop the SAME NUMBER.
So every filter arm is paired with a score-only arm cutting an identical
count. `filter - score` at matched count is the only number that means
anything here.

CALIBRATION DOES NOT TRANSFER, RANKING MIGHT. The filter is trained on
Zebrahub and applied to competition data, so its probability SCALE is
not to be trusted across the domain gap. Every arm therefore cuts by
PERCENTILE of the score within each dataset, not by an absolute
threshold -- only the ordering has to survive the transfer, which is the
weaker and more defensible assumption, and the one `morph_cheap`'s
rho-0.965 finding supports.

ANCHOR. The 0% arm changes nothing and must reproduce `biohub-poolsize`'s
D187 row: score 0.9386, adj_J 0.9254, div_J 0.1316, TP/FP/FN 5/26/7. It
uses the saved D187 pkl rather than retraining, so the ranker is the
submitted one byte for byte and 80 minutes of retraining are skipped. If
that arm drifts, nothing below is readable.

⚠ OFFLINE AUC HAS ANTI-PREDICTED THIS PIPELINE FOUR TIMES NOW
(scale-invariant edge features, the division model's four arms, D187
itself scoring LOWER on edge AUC while adj_J rose, and the div-gate
prune whose offline AP gain went monotone downhill in the pipeline).
+0.1418 offline is a reason to run this, not a result. **adj_J on the
holdout is the verdict.**
"""
from __future__ import annotations

import gc
import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-25a-node-filter"
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
                       list_datasets)

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
ARMS = (("BASE", 0.00, "filter"),
        ("F10", 0.10, "filter"),
        ("S10", 0.10, "score"),
        ("F20", 0.20, "filter"),
        ("S20", 0.20, "score"))
ACC_NAMES = ["acc_len", "acc_cos", "md_acc", "md-md_acc", "acc_hist"]
DIV_W = 0.55
MATCH_UM = 5.0


# ---- the D187 ranker, loaded not retrained -----------------------------
# biohub-poolsize saved it; this is the pkl behind submission 55757064,
# so the linker is the submitted one byte for byte and the 80 minutes of
# retraining buy nothing here -- the ranker is not what varies.
MIN_POOL = 100
_pk = None
for _pat in ("*/**/edge_ranker_d187.pkl",):
    for _p in pathlib.Path("/kaggle/input").glob(_pat):
        _pk = _p
        break
    if _pk:
        break
assert _pk is not None, ("edge_ranker_d187.pkl not found -- attach "
                         "faustine649/biohub-poolsize as a kernel source.")
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
_pool = _blob.get("pool") if isinstance(_blob, dict) else None
assert _pool is not None and len(_pool) >= MIN_POOL, (
    f"the pkl at {_pk} was trained on {0 if _pool is None else len(_pool)} "
    f"datasets; this run requires >= {MIN_POOL}, i.e. the D187 model.")
models = {"D187": _blob["model"]}
COLS = {"D187": _blob["cols"]}
print(f"  ranker loaded from {_pk}, pool {len(_pool)}", flush=True)


# ---- the node filter: trained on Zebrahub, dense GT, boxes only --------
NF_COLS = ["c_mean", "c_max", "c_std", "c_mass", "c_contrast", "c_centre"]
CROP_R = 3
NF_STRIDE = 5
# poolsize.py does not carry this -- it lives in cache_train.py, and the
# undefined-name pass caught its absence here exactly as it caught the
# same omission on 2026-08-24. SCALE is (1.625, 0.40625, 0.40625) and
# 1.625 / 0.40625 = 4, which is what makes the pooled grid isotropic.
# Detections are stored at FULL resolution in Y/X, so they divide by it.
POOL = 4
assert abs(SCALE[0] / SCALE[1] - POOL) < 1e-6, (
    f"SCALE {SCALE} does not imply a Y/X pooling factor of {POOL}; every "
    f"box statistic below would read the wrong voxels.")


def _pool_xy_nf(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm_nf(p):
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def box_stats(img, zyx, r=CROP_R):
    """node_bright's six columns, unchanged. The Y/X divide by POOL
    because detections are stored at FULL resolution there."""
    Z, Y, X = img.shape
    z0 = int(round(zyx[0]))
    y0 = int(round(zyx[1] / POOL))
    x0 = int(round(zyx[2] / POOL))
    c = img[max(0, z0 - r):min(Z, z0 + r + 1),
            max(0, y0 - r):min(Y, y0 + r + 1),
            max(0, x0 - r):min(X, x0 + r + 1)]
    if c.size == 0:
        return None
    lo = float(c.min())
    return (float(c.mean()), float(c.max()), float(c.std()),
            float(np.clip(c - lo, 0, None).sum()),
            float(c.max() - lo),
            float(img[min(z0, Z - 1), min(y0, Y - 1), min(x0, X - 1)]))


def _zmeta(zp):
    import json as _j
    with open(pathlib.Path(zp) / "0" / "zarr.json") as f:
        return _j.load(f)


def _zvol(zp, t, m):
    try:
        import zarr
        return np.asarray(zarr.open(str(zp), mode="r")["0"][t])
    except Exception:
        import blosc2
        buf = blosc2.decompress(
            open(pathlib.Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=np.dtype(m["data_type"])).reshape(
            tuple(m["shape"])[1:])


def comp_images(name):
    """Pooled, normalised competition volumes. Pure CPU: decompress plus
    a mean-pool, the same path morph_edge used."""
    zp = pathlib.Path(root) / "train" / f"{name}.zarr"
    if not zp.exists():
        warn_once("no_zarr", f"no zarr for {name} -> it CANNOT be "
                             "filtered and is passed through unchanged")
        return None
    m = _zmeta(zp)
    T = tuple(m["shape"])[0]
    present = [t for t in range(T)
               if (zp / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    T = max(present) + 1 if present else 0
    return [_norm_nf(_pool_xy_nf(_zvol(zp, t, m))) for t in range(T)]


print("\n=== training the node filter on Zebrahub (dense GT) ===",
      flush=True)
ZH = {f.stem[len("zhdet_"):]: f
      for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
ZCROPS = None
for _c in pathlib.Path("/kaggle/input").glob("*/**/crops"):
    ZCROPS = _c
    break
assert ZH and ZCROPS is not None, (
    "attach biohub-zh-detect and biohub-zebrahub-extract -- the filter "
    "cannot be trained without dense GT, and competition GT is sparse so "
    "'unmatched' there does not mean 'spurious'.")

nfX, nfy = [], []
_t0 = time.time()
for _k, _nm in enumerate(sorted(ZH)):
    _cf = ZCROPS / f"{_nm}.npz"
    if not _cf.exists():
        continue
    _zc = np.load(_cf)
    if "image" not in _zc.files:
        continue
    _img = _zc["image"]
    _zd = np.load(ZH[_nm])
    _nodes = np.asarray(_zd["nodes"], float)
    _T = min(_img.shape[0],
             max(int(kk[1:]) for kk in _zd.files if kk.startswith("c")) + 1)
    _gbt = defaultdict(list)
    for _r in range(len(_nodes)):
        _tt = int(_nodes[_r, 0])
        if 0 <= _tt < _T:
            _gbt[_tt].append(_r)
    for _t in range(0, _T, NF_STRIDE):
        _gi = _gbt.get(_t, [])
        if not _gi:
            continue
        _P = _norm_nf(_pool_xy_nf(np.asarray(_img[_t])))
        _cc = np.asarray(_zd[f"c{_t}"], float).reshape(-1, 3)
        _ss = np.asarray(_zd[f"s{_t}"], float).reshape(-1)
        _sel = _ss >= UNET_THRESH
        _cc, _ss = _cc[_sel], _ss[_sel]
        if len(_cc) == 0:
            continue
        _gt_um = _nodes[_gi, 1:] * SCALE[None, :]
        _det_um = _cc * SCALE[None, :]
        # greedy 1:1, node_bright v2's protocol
        _tree = cKDTree(_gt_um)
        _claimed = set()
        _kind = np.zeros(len(_cc), np.int8)
        for _i in np.argsort(-_ss):
            _near = _tree.query_ball_point(_det_um[_i], MATCH_UM)
            if not _near:
                continue
            _near.sort(key=lambda gg: float(
                np.linalg.norm(_gt_um[gg] - _det_um[_i])))
            _free = [gg for gg in _near if gg not in _claimed]
            if _free:
                _claimed.add(_free[0])
                _kind[_i] = 1
        for _i in range(len(_cc)):
            _b = box_stats(_P, _cc[_i])
            if _b is None:
                continue
            nfX.append((float(_ss[_i]),) + _b)
            nfy.append(int(_kind[_i]))
    del _zc, _img, _zd
    gc.collect()

nfX = np.asarray(nfX, np.float32)
nfy = np.asarray(nfy, np.int8)
print(f"  {len(nfX)} detections, {int(nfy.sum())} real "
      f"({100 * nfy.mean():.1f}%), {time.time() - _t0:.0f}s", flush=True)
assert len(nfX) > 10000, "too few rows to train a node filter"
NF = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                    max_leaf_nodes=31, random_state=0)
NF.fit(nfX, nfy)
print("  node filter trained on score + 6 box columns", flush=True)
del nfX, nfy
gc.collect()


# ---- filtering happens at LOAD time ------------------------------------
# Everything downstream -- seeds_of, pair_features, pass1_tracks, the
# ILP, the post-processing tail -- reads whatever load_dets returns, so
# dropping rows here IS "before the linker" with no further wiring.
_load_dets_raw = load_dets
CUT_FRAC = 0.0
CUT_BY = "filter"
_PSCORE = {}


def _pscores(name, dets):
    """p(real) per detection per frame, cached per dataset."""
    if name in _PSCORE:
        return _PSCORE[name]
    imgs = comp_images(name)
    out = []
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        if imgs is None or t >= len(imgs) or len(c) == 0:
            if imgs is not None and len(c):
                warn_once("frame_missing",
                          "a frame had no image -> its detections get "
                          "p=1 and are never cut, so that frame is "
                          "UNFILTERED rather than silently emptied")
            out.append(np.ones(len(c), np.float32))
            continue
        rows = []
        bad = 0
        for i in range(len(c)):
            b = box_stats(imgs[t], c[i])
            if b is None:
                bad += 1
                rows.append((float(s[i]), 0., 0., 0., 0., 0., 0.))
            else:
                rows.append((float(s[i]),) + b)
        if bad:
            warn_once("box_empty",
                      "a detection fell outside the pooled volume -> its "
                      "box columns are 0, which the filter reads as very "
                      "dim and will cut first")
        out.append(NF.predict_proba(
            np.asarray(rows, np.float32))[:, 1].astype(np.float32))
    _PSCORE[name] = out
    del imgs
    gc.collect()
    return out


def load_dets(name):
    dets = _load_dets_raw(name)
    if CUT_FRAC <= 0:
        return dets
    ps = _pscores(name, dets)
    out = []
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        m = s >= UNET_THRESH
        if m.sum() == 0:
            out.append((c, s))
            continue
        rank = ps[t] if CUT_BY == "filter" else s.astype(np.float32)
        thr = float(np.quantile(rank[m], CUT_FRAC))
        keep = (~m) | (rank > thr)
        out.append((c[keep], s[keep]))
    return out


# ---- STAGE B: the full pipeline, the only verdict that counts ----------
print(f"\n=== STAGE B: full pipeline on {len(HOLDOUT)} holdout datasets "
      f"(div_w {DIV_W}, global) ===", flush=True)
print("  BASE cuts NOTHING and must reproduce biohub-poolsize's D187 "
      "row: score 0.9386, adj_J 0.9254, div_J 0.1316, TP/FP/FN 5/26/7. "
      "If it does not, nothing else here is readable.", flush=True)
print("  Fxx cuts the bottom xx% by the node filter; Sxx cuts the bottom "
      "xx% by DETECTION SCORE alone. Fxx - Sxx at matched count is the "
      "only number that means anything -- dropping detections raises "
      "precision by construction.", flush=True)

res = {}
for arm, CUT_FRAC, CUT_BY in ARMS:
    PRED = pathlib.Path(f"/kaggle/working/accel_{arm.lower()}")
    PRED.mkdir(exist_ok=True)
    ta = time.time()
    for name in HOLDOUT:
        dets = load_dets(name)
        T = len(dets)
        seeds, vels, prev_of = pass1_tracks(dets)
        feats = pair_features(dets, None, T)
        probs = [models["D187"].predict_proba(rows[:, COLS["D187"]])[:, 1]
                 if len(pairs) else np.zeros(0)
                 for pairs, rows, P, C in feats]
        coords_rows, offset, n_glob = [], {}, 0
        for t in range(T):
            c, s = seeds[t]
            offset[t] = n_glob
            n_glob += len(c)
            for xyz in c:
                coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
        coords = np.asarray(np.round(coords_rows), np.int16)
        cand_edges = []
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            for (i, j), p in zip(pairs, probs[t]):
                if p >= ILP_FLOOR:
                    cand_edges.append((offset[t] + i, offset[t + 1] + j,
                                       float(p), 0.0))
        graph = PIPE.build_graph(coords, cand_edges)
        if graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                appearance_weight=0.0,
                disappearance_weight=ILP_DISAPP,
                division_weight=DIV_W)
            with PIPE.suppress_output():
                graph = solver.solve(graph)
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        nid_l = [int(v) for v in na["node_id"].to_list()]
        remap = {v: i for i, v in enumerate(nid_l)}
        nodes_arr = np.stack([np.asarray(na[c].to_list(), float)
                              for c in ("t", "z", "y", "x")], axis=1)
        rows_e, dropped = [], 0
        if graph.num_edges() > 0:
            ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
            for a, b in zip(ea["source_id"].to_list(),
                            ea["target_id"].to_list()):
                ra, rb = remap.get(int(a)), remap.get(int(b))
                if ra is None or rb is None:
                    dropped += 1
                    continue
                rows_e.append([ra, rb])
        if dropped:
            warn_once("ilp_endpoint_missing",
                      f"{dropped} solved edges had an endpoint absent from "
                      f"node_attrs and were dropped -> a LOWER BOUND "
                      f"(first in {name}, arm {arm})")
        e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
        nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                         "xyz": nodes_arr[i, 1:].astype(float)}
                 for i in range(len(nodes_arr))}
        edges = [(int(a) + 1, int(b) + 1) for a, b in e2]
        cand, cand_sc, cand_trees = [], [], []
        for t in range(T):
            c_all = np.asarray(dets[t][0], float)
            s_all = np.asarray(dets[t][1], float)
            cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
            cand.append(c_all[cm])
            cand_sc.append(s_all[cm])
            cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1])
                              else None)
        nid2 = len(nodes) + 1
        nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
        nodes, edges = _short_filter(nodes, edges)
        _linefit(nodes, edges)
        _add_safe_divisions(nodes, edges)
        ids = sorted(nodes)
        rm = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        e3 = np.array([[rm[a], rm[b]] for a, b in edges], int)
        dr._save(str(PRED / f"{name}.geff"), arr, e3)
        del dets, feats, probs, graph
        gc.collect()
    s = score_dir(PRED)
    res[arm] = s
    print(f"  {arm:<5} score {s['score']:.4f}  adj_J "
          f"{s['adj_edge_jaccard']:.4f}  div_J "
          f"{s['division_jaccard']:.4f}  TP/FP/FN "
          f"{s['division_tp']:.0f}/{s['division_fp']:.0f}/"
          f"{s['division_fn']:.0f}  ({time.time() - ta:.0f}s)", flush=True)

# v1 crashed on these two lines -- arm names left over from accel -- AFTER
# printing every number it was asked for. A summary line is not worth a
# 39-minute run; name the arms from ARMS so they cannot drift again.
print("\n  --- filter minus score, at MATCHED cut count ---", flush=True)
for _f, _s in (("F10", "S10"), ("F20", "S20")):
    if _f in res and _s in res:
        print(f"  {_f} - {_s}: score "
              f"{res[_f]['score'] - res[_s]['score']:+.4f}   adj_J "
              f"{res[_f]['adj_edge_jaccard'] - res[_s]['adj_edge_jaccard']:+.4f}"
              f"   div_J "
              f"{res[_f]['division_jaccard'] - res[_s]['division_jaccard']:+.4f}",
              flush=True)
print("\n  --- each arm against BASE ---", flush=True)
for _a in res:
    if _a == "BASE":
        continue
    print(f"  {_a} - BASE: score "
          f"{res[_a]['score'] - res['BASE']['score']:+.4f}   adj_J "
          f"{res[_a]['adj_edge_jaccard'] - res['BASE']['adj_edge_jaccard']:+.4f}",
          flush=True)
print("  Floor: holdout deltas under ~0.003 are not resolvable on the LB. "
      "adj_J is the one to read; it carries 10x div_J's weight.",
      flush=True)
print("  Offline AUC has anti-predicted this pipeline FOUR times. The "
      "+0.1418 that motivated this run is not evidence about adj_J; the "
      "rows above are.", flush=True)

warning_summary()
print("\nNODE FILTER DONE", flush=True)
