"""ranker_v3.py -- finish the edge re-ranker: deeper threshold sweep +
full-pipeline composite.

v2 (biohub-edge-ranker v2, attached as kernel source) established:
labels healthy (46.8% pos), AUC 0.9884, linker-only adj_J 0.9239 at
p_min 0.20 vs Hungarian 0.9185 (+0.0054), trend pointing lower. Its
output carries detection caches for all 44 datasets and the trained
model. This kernel: (1) rebuild training features from the caches and
retrain (fast, keeps label logic in one place); (2) sweep p_min
{0.05..0.25}; (3) run the FULL pipeline (ranker linking + snap gap
closing + safe divisions + short filter + linefit) at the best p_min ->
composite vs baseline 0.9318. No detection re-runs.
"""
from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-23x-zshare"
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


# ---------------- GATE_UM sweep ----------------------------------------
# Forensics found the binding constraint: 3 of 12 GT divisions have the
# parent AND both daughters detected, but the parent->daughter pair sits
# outside GATE_UM (10 um) so it never enters the candidate set at all.
# Widening the gate must come WITH a retrain: pair_features carries the
# candidate's rank and both candidate counts, whose distributions move
# with the gate.

def gt_of(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    child_of = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            child_of[ra].append(rb)
    by_t = defaultdict(list)
    for i in range(len(g_t)):
        by_t[g_t[i]].append(i)
    return g_t, g_pos, child_of, by_t


def _near(pool, x):
    if len(pool) == 0:
        return None, np.inf
    d = np.sqrt((((pool - x) * SCALE) ** 2).sum(1))
    k = int(np.argmin(d))
    return k, float(d[k])

GATE_SWEEP = (10.0, 14.0, 18.0)
ILP_WINDOW = 25
DIV_W = 0.55


def train_at_gate():
    """Rebuild the training set at the CURRENT GATE_UM and refit."""
    train_names = sorted(n for n in CACHES if n not in HOLDOUT)
    X_list, y_list = [], []
    t0 = time.time()
    for name in train_names:
        dets = load_dets(name)
        T = len(dets)
        g_t, g_pos_v, child_of, gt_by_t = gt_of(name)
        g_pos = g_pos_v * SCALE
        vels = pass1_velocities(dets)
        feats = pair_features(dets, vels, T)
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            if not pairs:
                continue
            Pu, Cu = P * SCALE[None, :], C * SCALE[None, :]

            def match(pos_u, idx_list):
                if not idx_list:
                    return {}
                gtree = cKDTree(g_pos[idx_list])
                m = {}
                for di, x in enumerate(pos_u):
                    dd, gi = gtree.query(x)
                    if dd <= MATCH_UM:
                        m[di] = idx_list[int(gi)]
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
                X_list.append(row)
                y_list.append(1 if j in j_true else 0)
    X = np.stack(X_list)
    y = np.asarray(y_list)
    perm = np.random.default_rng(0).permutation(len(y))
    n_val = max(1000, len(y) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(X[ti], y[ti])
    auc = roc_auc_score(y[vi], m.predict_proba(X[vi])[:, 1])
    print(f"  train: {len(y)} pairs, {y.mean():.3f} pos, AUC {auc:.4f}, "
          f"{time.time()-t0:.0f}s", flush=True)
    return m


def funnel(hold_g, clf_g):
    st = defaultdict(int)
    for name, (dets, feats, probs, T) in hold_g.items():
        g_t, g_pos, child_of, by_t = gt_of(name)
        for p, ch in child_of.items():
            if len(ch) < 2:
                continue
            t = int(g_t[p])
            if t + 1 >= T:
                continue
            P, _ = seeds_of(dets[t])
            C, _ = seeds_of(dets[t + 1])
            ip, dp = _near(P, g_pos[p])
            if ip is None or dp > MATCH_UM:
                st[0] += 1
                continue
            ics = []
            bad = False
            for c in ch[:2]:
                ic, dc = _near(C, g_pos[c])
                if ic is None or dc > MATCH_UM:
                    bad = True
                    break
                ics.append(ic)
            if bad:
                st[1] += 1
                continue
            pairs, rows, Pp, Cc = feats[t]
            pset = {(i, j): k for k, (i, j) in enumerate(pairs)}
            ks = [pset.get((ip, ic)) for ic in ics]
            if any(k is None for k in ks):
                st[2] += 1
                continue
            if min(float(probs[t][k]) for k in ks) < ILP_FLOOR:
                st[3] += 1
                continue
            st[4] += 1
    return st

GATE_XY = 10.0
GATE_Z = 10.0

# ---------------- division census ---------------------------------------
# zgate closed the gate line and named the real constraint: widening the
# gate moves a division from "outside gate" to "below floor" and REACHABLE
# never leaves 5. The ranker scores true long-z parent->daughter edges low
# because it cannot see what makes a division a division -- it scores each
# edge INDEPENDENTLY, while division is a property of the triple
# (parent, d1, d2): two children, roughly opposed, both continuing.
#
# A learned version of `_add_safe_divisions` needs divisions to train on.
# Detection caches cover 44 datasets; the train set has ~199. This counts

# ---------------- division geometry from GROUND TRUTH alone --------------
# Our z-share of 0.42 came from 26 divisions that the DETECTOR found. NMS
# (NMS_UM 4.0) merges nuclei that are close, and daughters separated along
# z are exactly the pairs most at risk -- so that 0.42 may be survivorship,
# not the population. Ground truth needs no detector, so this covers all
# 151 divisions in all 199 datasets and splits them by embryo.
#
# It settles three things at once:
#   1. per-embryo z-share (the two training embryos differ enormously:
#      1% vs 9% annotation sparsity, 26 vs 125 divisions)
#   2. GT z-share vs detected z-share -- the gap IS the selection bias
#   3. whether Zebrahub's 0.65-0.84 matches the competition's TRUE
#      distribution rather than its detected one

print("\n=== DIVISION GEOMETRY FROM GT (no detector involved) ===",
      flush=True)
names = sorted(list_datasets(root, "train"))
by_emb = defaultdict(list)
n_skip_edge = 0

for n in names:
    gp = geff_path(root, "train", n)
    if not gp.exists():
        print(f"  WARN {n}: no geff -- skipped", flush=True)
        continue
    gt = read_geff(gp)
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    emb = n.split("_")[0]
    for p, ch in kids.items():
        if len(ch) < 2:
            continue
        c1, c2 = ch[0], ch[1]
        if g_t[c1] != g_t[p] + 1 or g_t[c2] != g_t[p] + 1:
            n_skip_edge += 1
            continue
        v1 = (g_pos[c1] - g_pos[p]) * SCALE
        v2 = (g_pos[c2] - g_pos[p]) * SCALE
        sep = float(np.linalg.norm((g_pos[c1] - g_pos[c2]) * SCALE))
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        far, v = (n1, v1) if n1 >= n2 else (n2, v2)
        ang = np.nan
        if n1 > 1e-6 and n2 > 1e-6:
            ang = float(np.degrees(np.arccos(
                np.clip(float(v1 @ v2) / (n1 * n2), -1, 1))))
        by_emb[emb].append((sep, far, abs(float(v[0])) / max(far, 1e-9),
                            ang, abs(float(v[0])),
                            float(np.linalg.norm(v[1:]))))

if n_skip_edge:
    print(f"  NOTE {n_skip_edge} divisions whose daughters are not at t+1 "
          f"were skipped (the metric only scores t->t+1 edges)", flush=True)


def rep(tag, rows):
    if not rows:
        print(f"  {tag}: no divisions", flush=True)
        return
    a = np.array([r[:3] for r in rows], float)
    ang = np.array([r[3] for r in rows], float)
    dz = np.array([r[4] for r in rows], float)
    dxy = np.array([r[5] for r in rows], float)
    ang = ang[~np.isnan(ang)]
    print(f"  {tag}: n={len(rows)}", flush=True)
    print(f"    sister separation um : p50 {np.median(a[:,0]):5.2f}  "
          f"p10 {np.quantile(a[:,0],.1):5.2f}  "
          f"p90 {np.quantile(a[:,0],.9):5.2f}", flush=True)
    print(f"    worst parent->dau um : p50 {np.median(a[:,1]):5.2f}  "
          f"p90 {np.quantile(a[:,1],.9):5.2f}  "
          f"max {a[:,1].max():5.2f}", flush=True)
    print(f"    **z share**          : p50 {np.median(a[:,2]):5.2f}  "
          f"mean {a[:,2].mean():5.2f}  "
          f"p10 {np.quantile(a[:,2],.1):5.2f} "
          f"p90 {np.quantile(a[:,2],.9):5.2f}", flush=True)
    print(f"    that split, um       : |dz| p50 {np.median(dz):5.2f}   "
          f"|dxy| p50 {np.median(dxy):5.2f}", flush=True)
    if len(ang):
        print(f"    daughter angle deg   : p50 {np.median(ang):5.0f}  "
              f"p90 {np.quantile(ang,.9):5.0f}", flush=True)
    for g in (10, 12, 14):
        print(f"    a {g:2d} um gate admits {100.0*(a[:,1]<=g).mean():3.0f}%",
              flush=True)


print(f"\n  embryos present: {sorted(by_emb)}", flush=True)
for emb in sorted(by_emb):
    print("", flush=True)
    rep(f"embryo {emb}", by_emb[emb])
print("", flush=True)
allr = [r for v in by_emb.values() for r in v]
rep("ALL competition divisions", allr)

print("\n=== THE THREE COMPARISONS ===", flush=True)
if allr:
    z_gt = float(np.median([r[2] for r in allr]))
    print(f"  1. GT z-share {z_gt:.2f}  vs  detected z-share 0.42 "
          f"(census, 26 divisions the detector found)", flush=True)
    if z_gt > 0.55:
        print(f"     -> GT is HIGHER. The detected sample is biased: "
              f"z-separated daughters are being lost before we ever see "
              f"them (NMS_UM 4.0 is the suspect).", flush=True)
    else:
        print(f"     -> GT agrees with the detected sample, so 0.42 is "
              f"real and NOT a selection artefact.", flush=True)
    print(f"  2. competition {z_gt:.2f}  vs  Zebrahub 0.65-0.84 "
          f"(five movies, 468k validated divisions)", flush=True)
    zs = {e: float(np.median([r[2] for r in v])) for e, v in by_emb.items()}
    if len(zs) > 1:
        sp = max(zs.values()) - min(zs.values())
        print(f"  3. embryo-to-embryo spread within the competition: "
              f"{sp:.2f} ({zs})", flush=True)
        print(f"     -> compare that spread against the "
              f"competition-vs-Zebrahub gap. If it is of the same order, "
              f"Zebrahub is just another embryo, not another world.",
              flush=True)
print("\nZSHARE DONE", flush=True)
