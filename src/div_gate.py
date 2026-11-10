"""div_gate.py -- put the division model INTO the pipeline and
measure div_J.

Everything before this was offline AUC/AP. This is the real metric.

What the recorded pipeline does at div_w 0.50 (progress.md fork table):
590 forks reach the output, only 148 from the ILP -- the other 442 come
from the _add_safe_divisions heuristic. The metric counts 5 TP against 26
FP, so div_J is 0.1316 and the division term contributes 0.0132 of the
0.9345 score. Divisions have never been worth taking because BASE buys
them at 6.9 false ones per true one, right on the metric's ~7 budget.

biohub-div-kinematics measured a division model that buys them at 1.8.
This kernel spends that: solve each holdout dataset once exactly as the
recorded pipeline does, score every resulting fork with the FULL division
model, then sweep a prune threshold that removes the least convincing
forks and rescore.

Pruning only ever REMOVES a second daughter, so it cannot break graph
consistency, and frac 0.00 prunes nothing -- that row must reproduce the
recorded numbers, and it is the anchor for every other row.

Forks whose parent or daughters are synthetic gap-close nodes cannot be
scored and are NEVER pruned, at any threshold.

Not covered here: inserting divisions the solver declined (the 2 stuck
below ILP_FLOOR in the forensics funnel). Prune first -- FPs outnumber TPs
5:1, so precision is where the mass is.
"""
from __future__ import annotations

import gc
import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-24c-div-gate"
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




# ---- the trajectory, BEFORE and AFTER the candidate division -----------
# User's point, 2026-08-24: a division is a physical event that perturbs
# motion, and motion is a DERIVATIVE -- it does not exist inside one frame.
# triple_feats carries exactly ONE temporal number, the mother's speed
# MAGNITUDE (col 19), from a single previous transition, direction thrown
# away by np.linalg.norm.
#
# Two directions in time, and the user was right to push back on doing
# only the cheap one:
#   BACKWARD  the mother's own past. Free and unambiguous -- before t she
#             is ONE cell. But mitotic rounding is brief, and if the frame
#             interval is coarse relative to it the slow-down may not be
#             sampled at all.
#   FORWARD   what the two daughters then do. Costs a forward chain and is
#             noisier, but the divergence after a split lasts several
#             frames, so it is far more likely to be sampled.
# Not a leak: this is offline tracking over a finished movie and the
# solver already optimises globally over all frames.

HIST_K = 5          # frames of mother history to walk back
FWD_K = 5           # frames of daughter future to walk forward
N_HIST = 10
N_FWD = 10
N_COUPLE = 4
NFEAT_BASE = 25     # triple_feats, unchanged
OFF_HIST = NFEAT_BASE
OFF_FWD = OFF_HIST + N_HIST
OFF_COUPLE = OFF_FWD + N_FWD
NFEAT_FULL = OFF_COUPLE + N_COUPLE      # 49
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
    """Velocities AND both link maps, from one greedy pass.

    prev_of[t][i] = index in frame t-1 linked to cell i at t, or -1.
    next_of[t][i] = index in frame t+1 linked to cell i at t, or -1.
    pass1_velocities discards both; the temporal features need them."""
    T = len(dets)
    seeds = [seeds_of(dets[t]) for t in range(T)]
    vels = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    prev_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    next_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
            prev_of[t + 1][ci] = pi
            next_of[t][pi] = ci
    return seeds, vels, prev_of, next_of


def _walk(seeds, link, t0, i0, k, step):
    """Follow a cell through `link` for up to k frames. step is -1 to go
    back, +1 to go forward. Returns (pts_um, scores) with element 0 at
    t0 itself."""
    pts, scs = [], []
    tt, ii = t0, int(i0)
    T = len(seeds)
    while 0 <= tt < T and ii >= 0 and len(pts) <= k:
        P, ps = seeds[tt]
        if ii >= len(P):
            break
        pts.append(P[ii] * SCALE)
        scs.append(float(ps[ii]))
        nt = tt + step
        if not (0 <= nt < T):
            break
        ii = int(link[tt][ii])
        tt = nt
    return np.asarray(pts, float), np.asarray(scs, float)


def _angle_deg(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < EPS or nv < EPS:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _steps(pts):
    """Consecutive displacement vectors along pts, in list order."""
    return [pts[m + 1] - pts[m] for m in range(len(pts) - 1)]


def hist_feats(pts, scs):
    """N_HIST columns from the mother's own past. pts is NEWEST FIRST.

    Column 0 is the number of steps actually available; when it is 0 the
    rest are zeros, so the model can split on column 0 and treat 'no
    history' as its own regime rather than as 'a stationary cell'."""
    n = len(pts)
    if n < 2:
        warn_once("hist_absent",
                  "a mother had no backward link (frame 0, or the greedy "
                  "pass left it unlinked) -> all 10 history columns are 0 "
                  "and hist_len says so; they are NOT measurements of a "
                  "stationary cell")
        return [0.0] * N_HIST
    steps = [pts[m] - pts[m + 1] for m in range(n - 1)]   # newest first
    norms = np.array([float(np.linalg.norm(s)) for s in steps])
    v_last = float(norms[0])
    v_mean = float(norms.mean())
    path = float(norms.sum())
    net = float(np.linalg.norm(pts[0] - pts[-1]))
    if len(norms) >= 2:
        # positive slope = faster further in the past = slowing toward t
        v_slope = float(np.polyfit(np.arange(len(norms)), norms, 1)[0])
        curv_last = _angle_deg(steps[0], steps[1])
        curv_mean = float(np.mean([_angle_deg(steps[m], steps[m + 1])
                                   for m in range(len(steps) - 1)]))
    else:
        warn_once("hist_one_step",
                  "a mother had only ONE backward step, so slope and "
                  "curvature are undefined -> substituting 0.0 for those "
                  "three columns; hist_len distinguishes this case")
        v_slope = curv_last = curv_mean = 0.0
    units = np.array([s / max(float(np.linalg.norm(s)), EPS) for s in steps])
    dir_stab = float(np.linalg.norm(units.mean(axis=0)))
    sc_hist = float(np.mean(scs[1:])) if len(scs) > 1 else float(scs[0])
    return [float(len(steps)), v_last, v_mean, v_last / max(v_mean, EPS),
            v_slope, float(net / max(path, EPS)), curv_last, curv_mean,
            dir_stab, float(scs[0]) / max(sc_hist, EPS)]


def fwd_feats(pa, sa, pb, sb, sep0):
    """N_FWD columns from what the two daughters DO next.

    The claim being tested: a real pair both persists and keeps drifting
    apart, symmetrically; a spurious pair has one member vanish, or the
    two converge again because they were simply passing each other.

    Column 0 is how many steps BOTH survived; 0 means the rest are
    zeros."""
    na, nb = len(pa), len(pb)
    surv = min(na, nb) - 1
    if surv < 1:
        warn_once("fwd_absent",
                  "a candidate daughter had no forward link (last frame, "
                  "or the greedy pass dropped it) -> the 8 trajectory "
                  "columns are 0 and surv_both says so; they are NOT "
                  "measurements of a stable pair")
        # surv_gap survives the early return on purpose: ONE member
        # vanishing while the other persists is the strongest spurious
        # signature there is, and zeroing it would throw that away.
        return [0.0, float(abs(na - nb))] + [0.0] * (N_FWD - 2)
    a, b = pa[:surv + 1], pb[:surv + 1]
    seps = np.array([float(np.linalg.norm(a[m] - b[m]))
                     for m in range(len(a))])
    s0 = max(float(sep0), EPS)
    if len(seps) >= 2:
        sep_slope = float(np.polyfit(np.arange(len(seps)), seps, 1)[0])
    else:
        sep_slope = 0.0
    da = a[-1] - a[0]
    db = b[-1] - b[0]
    nda, ndb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    cos_dirs = (float(np.dot(da, db)) / (nda * ndb)
                if nda > EPS and ndb > EPS else 0.0)
    cen = 0.5 * (a + b)
    csteps = _steps(cen)
    cpath = float(sum(float(np.linalg.norm(s)) for s in csteps))
    cnet = float(np.linalg.norm(cen[-1] - cen[0]))
    ma = float(np.mean(sa[:surv + 1]))
    mb = float(np.mean(sb[:surv + 1]))
    return [float(surv), float(abs(na - nb)),
            float(seps[-1] / s0), float(seps.min() / s0), sep_slope,
            cos_dirs, min(ma, mb), min(ma, mb) / max(max(ma, mb), EPS),
            cpath / max(len(csteps), 1),
            cnet / max(cpath, EPS)]


def couple_feats(pts, a_um, b_um, p_um):
    """N_COUPLE columns tying the mother's history to the candidate pair.

    The physical claim: at a real division the CENTROID of the two
    daughters carries the mother's momentum forward; a spurious pair's
    centroid is dragged off it."""
    mid = 0.5 * (a_um + b_um)
    sep = float(np.linalg.norm(a_um - b_um))
    if len(pts) < 2:
        return [float(np.linalg.norm(mid - p_um)), 0.0, 0.0, 0.0]
    steps = [pts[m] - pts[m + 1] for m in range(len(pts) - 1)]
    norms = np.array([float(np.linalg.norm(s)) for s in steps])
    v_mean = float(norms.mean())
    resid = float(np.linalg.norm(mid - (p_um + steps[0])))
    heading = np.mean([s / max(float(np.linalg.norm(s)), EPS)
                       for s in steps], axis=0)
    axis = a_um - b_um
    na, nh = float(np.linalg.norm(axis)), float(np.linalg.norm(heading))
    axis_vs_motion = (abs(float(np.dot(axis, heading)) / (na * nh))
                      if na > EPS and nh > EPS else 0.0)
    return [resid, resid / max(v_mean, EPS), axis_vs_motion,
            sep / max(v_mean, EPS)]


def triple_feats_ext(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T,
                     seeds, prev_of, next_of):
    """25 base columns EXACTLY as triple_feats computes them, then 10
    backward, 10 forward, 4 coupling. One build, four column views -- the
    arms can differ ONLY in what the model is allowed to see."""
    base = triple_feats(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T)
    hp, hs = _walk(seeds, prev_of, t, i, HIST_K, -1)
    fa, sa = _walk(seeds, next_of, t + 1, ja, FWD_K, +1)
    fb, sb = _walk(seeds, next_of, t + 1, jb, FWD_K, +1)
    a_um, b_um = C[ja] * SCALE, C[jb] * SCALE
    sep0 = float(np.linalg.norm(a_um - b_um))
    return (base + hist_feats(hp, hs) + fwd_feats(fa, sa, fb, sb, sep0)
            + couple_feats(hp, a_um, b_um, P[i] * SCALE))


def build(dets, T, div_set, tag, rng=None):
    """Same gating, same labels, same negative thinning as div_train's
    build -- only the feature row is longer.

    div_set: {(t, i_parent_det, j1_det, j2_det)} known true divisions."""
    X, y, meta = [], [], []
    seeds, vels, prev_of, next_of = pass1_tracks(dets)
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) < 2 or len(C) < 2:
            continue
        Pu, Cu = P * SCALE[None, :], C * SCALE[None, :]
        tree = cKDTree(Cu)
        for i in range(len(P)):
            idx = tree.query_ball_point(Pu[i], GATE_UM)
            if len(idx) < 2:
                continue
            idx = sorted(idx, key=lambda j: np.linalg.norm(Cu[j] - Pu[i]))
            idx = [j for j in idx if cs[j] >= MIN_SCORE_A][:TOPK]
            if len(idx) < 2:
                continue
            sp = spacing_at(Pu, Pu[i])
            truth = {tuple(sorted(v[2:])) for v in div_set if v[0] == t
                     and v[1] == i}
            for m in range(len(idx)):
                for n in range(m + 1, len(idx)):
                    ja, jb = idx[m], idx[n]
                    lab = 1 if tuple(sorted((ja, jb))) in truth else 0
                    if lab == 0 and rng is not None and rng.random() > KEEP:
                        continue
                    X.append(triple_feats_ext(
                        P, C, len(P), i, ja, jb, ps, cs,
                        vels[t][i] if len(vels[t]) > i else None,
                        sp, t, T, seeds, prev_of, next_of))
                    y.append(lab)
                    meta.append(tag)
    return X, y, meta




# ---- load the two trained models from kernel inputs --------------------
def _find(pattern):
    hits = sorted(pathlib.Path("/kaggle/input").glob(pattern))
    if not hits:
        hits = sorted(pathlib.Path("/kaggle/input").glob("*/**/" + pattern))
    return hits[0] if hits else None


p_edge = _find("edge_ranker.pkl")
p_div = _find("div_model_full.pkl")
assert p_edge is not None, "attach biohub-ranker-fine (edge_ranker.pkl)"
assert p_div is not None, "attach biohub-div-kinematics (div_model_full.pkl)"
with open(p_edge, "rb") as fh:
    _e = pickle.load(fh)
clf_edge = _e["model"] if isinstance(_e, dict) else _e
with open(p_div, "rb") as fh:
    _d = pickle.load(fh)
clf_div, DIV_COLS = _d["model"], _d["cols"]
print(f"edge ranker: {p_edge}", flush=True)
print(f"div model:   {p_div}  ({len(DIV_COLS)} cols, "
      f"{_d.get('version')})", flush=True)
assert len(DIV_COLS) == NFEAT_FULL, \
    f"div model wants {len(DIV_COLS)} cols, this build makes {NFEAT_FULL}"

DIV_W = 0.55            # the recorded best; global solve, no windowing
PRUNE_FRACS = (0.0, 0.25, 0.50, 0.75, 0.90)


# ---- solve each holdout dataset ONCE, then score every fork ------------
print(f"\n=== SOLVING {len(HOLDOUT)} HOLDOUT DATASETS "
      f"(div_w {DIV_W}, global) ===", flush=True)
SOLVED = {}
all_pdiv = []
t0 = time.time()
for k_ds, name in enumerate(HOLDOUT):
    dets = load_dets(name)
    T = len(dets)
    seeds, vels, prev_of, next_of = pass1_tracks(dets)
    feats = pair_features(dets, pass1_velocities(dets), T)
    probs = [clf_edge.predict_proba(rows)[:, 1] if len(pairs) else
             np.zeros(0) for pairs, rows, P, C in feats]

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
                  f"node_attrs and were dropped -> those datasets' scores "
                  f"are a LOWER BOUND (first seen in {name})")
    e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)

    nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                     "xyz": nodes_arr[i, 1:].astype(float)}
             for i in range(len(nodes_arr))}
    edges = [(int(a) + 1, int(b) + 1) for a, b in e2]

    # node -> (frame, detection index), taken BEFORE _gap_close and
    # _linefit run. The solver does not preserve coords row order, so the
    # mapping goes through the integer coordinates, which are exact at
    # this point; after _linefit they are not, which is why this block
    # cannot be moved later.
    det_of_node = {}
    key_to_det = {}
    for t in range(T):
        c, s = seeds[t]
        for di, xyz in enumerate(c):
            r = np.round([t, xyz[0], xyz[1], xyz[2]]).astype(int)
            key_to_det[(int(r[0]), int(r[1]), int(r[2]), int(r[3]))] = \
                (t, di)
    for nid, nd in nodes.items():
        r = np.round([nd["t"], *nd["xyz"]]).astype(int)
        hit = key_to_det.get((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
        if hit is not None:
            det_of_node[nid] = hit

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

    # ---- score every fork with the division model ----------------------
    out_of = defaultdict(list)
    for a, b in edges:
        out_of[a].append(b)
    fork_p = {}
    unscorable = 0
    for a, kids in out_of.items():
        if len(kids) < 2:
            continue
        pa = det_of_node.get(a)
        if pa is None:
            unscorable += 1
            continue
        t, i = pa
        ds = [det_of_node.get(b) for b in kids]
        ds = [d for d in ds if d is not None and d[0] == t + 1]
        if len(ds) < 2:
            unscorable += 1
            continue
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        ja, jb = ds[0][1], ds[1][1]
        row = triple_feats_ext(P, C, len(P), i, ja, jb, ps, cs,
                               vels[t][i] if len(vels[t]) > i else None,
                               spacing_at(P * SCALE[None, :],
                                          P[i] * SCALE),
                               t, T, seeds, prev_of, next_of)
        fork_p[a] = float(clf_div.predict_proba(
            np.asarray([row], np.float32)[:, DIV_COLS])[:, 1][0])
        all_pdiv.append(fork_p[a])
    if unscorable:
        warn_once("fork_unscorable",
                  f"a fork's parent or daughters were synthetic gap-close "
                  f"nodes, or the daughters were not both at t+1, so the "
                  f"division model could not see it -> such forks are "
                  f"NEVER pruned, at any threshold (first in {name})")
    SOLVED[name] = (nodes, edges, fork_p, len(out_of))
    if (k_ds + 1) % 4 == 0 or k_ds == len(HOLDOUT) - 1:
        print(f"  {k_ds + 1}/{len(HOLDOUT)} solved "
              f"({time.time() - t0:.0f}s)", flush=True)

n_forks = sum(1 for v in SOLVED.values() for _ in v[2])
print(f"\n  {n_forks} scorable forks across the holdout; "
      f"p_div median {np.median(all_pdiv):.4f}, "
      f"p90 {np.quantile(all_pdiv, 0.9):.4f}", flush=True)
warning_summary()


# ---- sweep: prune the least convincing forks, then score ---------------
print("\n=== PRUNE SWEEP (drop the weakest forks by p_div) ===",
      flush=True)
print("  frac 0.00 prunes nothing and reproduces the recorded pipeline: "
      "score 0.9345, adj_J 0.9214, div_J 0.1316, div TP/FP/FN 5/26/7 "
      "at div_w 0.50. This run uses div_w 0.55.", flush=True)

results = {}
for frac in PRUNE_FRACS:
    thr = (-1.0 if frac <= 0 else float(np.quantile(all_pdiv, frac)))
    PRED = pathlib.Path(f"/kaggle/working/prune_{int(frac * 100):02d}")
    PRED.mkdir(exist_ok=True)
    n_cut = 0
    for name, (nodes0, edges0, fork_p, _) in SOLVED.items():
        edges = list(edges0)
        if thr > 0:
            out_of = defaultdict(list)
            for a, b in edges:
                out_of[a].append(b)
            drop = set()
            for a, kids in out_of.items():
                if len(kids) < 2:
                    continue
                p = fork_p.get(a)
                if p is None or p >= thr:
                    continue
                # keep the daughter nearer the parent; the far one is the
                # one a spurious fork usually invents
                pa = nodes0[a]["xyz"]
                far = max(kids, key=lambda b: _dist_um(pa, nodes0[b]["xyz"]))
                drop.add((a, far))
                n_cut += 1
            edges = [(a, b) for a, b in edges if (a, b) not in drop]
        nodes2, edges2 = _short_filter(dict(nodes0), edges)
        ids = sorted(nodes2)
        rm = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes2[g]["t"], *nodes2[g]["xyz"]] for g in ids],
                       float)
        e3 = np.array([[rm[a], rm[b]] for a, b in edges2], int)
        dr._save(str(PRED / f"{name}.geff"), arr, e3)
    s = score_dir(PRED)
    results[frac] = (s, n_cut, thr)
    div_bits = " ".join(f"{k} {s[k]:.4f}" for k in sorted(s)
                        if "div" in k.lower())
    print(f"  prune {frac:.0%} (p_div < {thr:.4f}, cut {n_cut:>4} forks): "
          f"score {s['score']:.4f}  adj_J {s['adj_edge_jaccard']:.4f}  "
          f"{div_bits}", flush=True)

print("\n  available metric keys:", sorted(results[0.0][0].keys()),
      flush=True)
best = max(results, key=lambda f: results[f][0]["score"])
b0 = results[0.0][0]["score"]
print(f"\n  BEST prune {best:.0%} -> score {results[best][0]['score']:.4f} "
      f"vs {b0:.4f} unpruned  ({results[best][0]['score'] - b0:+.4f})",
      flush=True)
print("  The project's floor: holdout deltas under ~0.003 are not "
      "resolvable on the LB.", flush=True)

warning_summary()
print("\nDIV GATE DONE", flush=True)
