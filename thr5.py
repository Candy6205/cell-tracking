"""Re-test the detection gate in THREE domains. Is +0.0033 a 6bba artefact?

SCRIPT_VERSION 2026-08-26a-thr5

WHY THIS IS BEING RE-RUN. `biohub-thr-sweep` measured that lowering the
detection gate from 0.15 to 0.12 wins on all three terms -- score
+0.0036, adj_J +0.0033, div_J +0.0036 -- and that is the only positive
result the session produced. It was measured on the 12 `6bba` holdout
datasets and nothing else.

Hours later the leaderboard answered a different question and made this
one urgent. **The local holdout is now 0 for 2 against the LB:**

| change | local delta | LB delta |
|---|---|---|
| windowed vs global | +0.0019 | -0.001 |
| ranker pool 32 -> 187 | **+0.0073** | -0.001 |

The second was reproduced bit for bit by two runs with both anchors
holding, and the hidden-test output differed by 0.07%. **+0.0033 sits
inside the range that has already failed twice.**

The user's correction, and it is the reason this run exists: if the
holdout inverts on gains it has no claim to reliability on losses
either. The honest boundary is a MAGNITUDE, not a direction -- the
holdout can reject an effect much larger than the ~0.007 inversion scale
and can neither reject nor select anything near 0.003.

THE THREE DOMAINS, and why Zebrahub belongs in a standing criterion:

    6bba      the 12 holdout, clean of the ranker
    44b6      the OTHER imaging batch -- 327 cells/frame against 97,
              displacement p99 4.90 um against 7.81
    zebrahub  a different instrument. **No public model has seen it**,
              which `6bba` cannot claim: DeepCenter trained on 44b6 and
              validated on 6bba, and the xl detector's split is unknown.

WHY D32 AND NOT D187. D187 trained on 187 of the 199 competition
datasets, so EVERY 44b6 dataset is in its training pool and its edge
probabilities there are optimistic for the candidate set built at the
shipped threshold -- a bias pointing straight at the null result. D32
trained on 32, so 44b6 datasets outside that pool are clean. D32 is also
the ranker in the LB 0.921 submission, which makes this the sweep on the
pipeline that actually ships.

PRE-REGISTERED RULE, unchanged from `biohub-post-sweep`: **a threshold
counts as better only if its adj_J delta has the SAME SIGN in all three
domains.** If 0.12 wins on 6bba alone, it is a 6bba artefact and belongs
with poolsize.

NOTE ON COST. Unlike the post-link sweep, changing the gate changes the
DETECTIONS, so the ILP cannot be solved once and reused -- every
threshold is a full solve. That is why this sweeps four values and not
fifteen.
"""
from __future__ import annotations

import gc
import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-26a-thr5"
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
SHORT_MIN = 6            # REBOUND per configuration
LINEFIT_WEIGHT = 0.8     # REBOUND per configuration
LINEFIT_WINDOW = 2       # REBOUND per configuration
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
GAP1_GATE_UM = 9.0       # REBOUND per configuration
GAP1_SNAP_UM = 3.2       # REBOUND per configuration
GAP1_MIN_CAND_SCORE = 0.10   # REBOUND per configuration
GAP1_CAP_FRAC = 0.003    # REBOUND per configuration
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


def _with_ratio(rows):
    """summarise(), plus the node-count columns it does not aggregate.

    `summarise` returns edge_jaccard and adj_edge_jaccard but drops
    `total_node_ratio`, which is the entire difference between them:
        adj = J * (1 - 0.1 * total_node_ratio)
    Without it a sweep cannot say whether a change improved the EDGES or
    merely the NODE COUNT, and every sweep in this project so far read
    only the product. Weighted by tp+fp+fn, the same weights
    adj_edge_jaccard uses, so the three columns are consistent.

    Purely ADDITIVE: summarise's own keys are untouched.
    """
    out = dict(summarise(rows))
    good = [r for r in rows
            if r.get("total_node_ratio") == r.get("total_node_ratio")
            and r["edge_tp"] == r["edge_tp"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in good]
    tw = sum(w)
    out["total_node_ratio"] = (
        sum(r["total_node_ratio"] * wi for r, wi in zip(good, w)) / tw
        if tw else float("nan"))
    return out


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
    return _with_ratio(rows)


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


# ---- the D187 ranker, loaded not retrained -----------------------------
# D187 ON PURPOSE: it is what the submission runs. thr3 used D32 to keep
# the 44b6 arm out of sample, and that half of the rule is sound -- but
# it moves the measurement onto a model poolsize measured at +0.0032
# adj_J away from the shipped one, to dodge a bias whose DIRECTION was
# never established. The leaderboard runs D187.
MIN_POOL = 100
_pk = None
for _p in pathlib.Path("/kaggle/input").glob("*/**/edge_ranker_d187.pkl"):
    _pk = _p
    break
assert _pk is not None, ("edge_ranker_d187.pkl not found -- attach "
                         "faustine649/biohub-poolsize as a kernel source.")
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
_pool = _blob.get("pool") if isinstance(_blob, dict) else None
assert _pool is not None and len(_pool) >= MIN_POOL, (
    f"the pkl at {_pk} was trained on {0 if _pool is None else len(_pool)} "
    f"datasets; this run needs the LARGE pool (D187), the model the "
    f"submission actually uses.")
MODEL = _blob["model"]
MCOLS = _blob["cols"]
RANKER_POOL = set(_pool)
print(f"  ranker loaded from {_pk}, pool {len(_pool)} (D187, the model "
      f"the submission runs)", flush=True)
print("  \u26a0 EVERY 44b6 dataset is inside this ranker's training pool. "
      "That arm is therefore IN-SAMPLE and reports FIT, not transfer. "
      "The independent domains here are 6bba and Zebrahub -- TWO, not "
      "three. Do not count this run as satisfying the three-domain "
      "rule.", flush=True)


# ---- the three evaluation domains --------------------------------------
# 6bba is poolsize's holdout verbatim. 44b6 is chosen AT RUNTIME from
# whatever caches are mounted, sorted, so no dataset name is invented
# here and a missing mount fails loudly instead of silently shrinking the
# comparison.
N_44B6 = 12
SET_6BBA = [n for n in HOLDOUT if n in CACHES]
# thr3 selected 44b6 datasets NOT in D32's pool. With D187 there are
# none outside the pool, so filtering would empty the arm. Keep the same
# twelve thr3 used -- sorted, excluding D32's sixteen -- so the two runs
# describe the same datasets and only the ranker and SHORT_MIN differ.
SET_44B6 = sorted(n for n in CACHES if n.startswith("44b6_"))[:N_44B6]
_in_pool = sum(1 for n in SET_44B6 if n in RANKER_POOL)
print(f"  44b6: {len(SET_44B6)} datasets, {_in_pool} of them INSIDE the "
      f"ranker's training pool", flush=True)
if _in_pool:
    print(f"  ⚠ This arm is IN-SAMPLE for D187 and reports FIT, not "
          f"transfer. thr3 avoided that by using D32 and picking the "
          f"twelve 44b6 datasets outside its pool; on D187 there is no "
          f"such subset, so the choice is between measuring the shipped "
          f"model with a compromised arm and measuring a model we do "
          f"not ship. The user chose the former. The INDEPENDENT "
          f"domains in this run are 6bba and Zebrahub -- TWO.",
          flush=True)
assert len(SET_6BBA) == len(HOLDOUT), (
    f"only {len(SET_6BBA)} of {len(HOLDOUT)} holdout caches are mounted; "
    f"the 6bba row would not be comparable to the recorded anchor")
assert len(SET_44B6) == N_44B6, (
    f"only {len(SET_44B6)} 44b6 caches are mounted, need {N_44B6}; without "
    f"the second imaging batch this run cannot test cross-batch "
    f"reproduction, which is its whole purpose")
print(f"  domains: 6bba {len(SET_6BBA)}, 44b6 {len(SET_44B6)}", flush=True)

ZH = {f.stem[len("zhdet_"):]: f
      for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
N_ZH = 8
SET_ZH = sorted(ZH)[:N_ZH]
if not SET_ZH:
    print("  WARN biohub-zh-detect is not attached -> the zebrahub domain "
          "is ABSENT and the three-domain rule degrades to two. Say so in "
          "the write-up rather than reporting a two-domain result as if it "
          "satisfied the rule.", flush=True)
else:
    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)


def zh_load(name):
    """(dets, T, gt_nodes, gt_edges, n_cross) for one Zebrahub crop.
    Carried over from zh_pipeline.py unchanged."""
    z = np.load(ZH[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]
    nodes = np.asarray(z["nodes"], float)
    tid = np.asarray(z["track_id"], np.int64)
    pid = np.asarray(z["parent_track_id"], np.int64)
    keep = (nodes[:, 0] >= 0) & (nodes[:, 0] < T)
    nodes, tid, pid = nodes[keep], tid[keep], pid[keep]
    order = np.lexsort((nodes[:, 0], tid))
    nodes, tid, pid = nodes[order], tid[order], pid[order]
    row_of = {(int(tid[i]), int(nodes[i, 0])): i for i in range(len(nodes))}
    edges = []
    for i in range(1, len(nodes)):
        if tid[i] == tid[i - 1] and nodes[i, 0] == nodes[i - 1, 0] + 1:
            edges.append((i - 1, i))
    first, par_of = {}, {}
    for i in range(len(nodes)):
        k = int(tid[i])
        if k not in first or nodes[i, 0] < nodes[first[k], 0]:
            first[k] = i
        if pid[i] > 0:
            par_of[k] = int(pid[i])
    n_cross = 0
    for k, i in first.items():
        p = par_of.get(k)
        if p is None:
            continue
        j = row_of.get((p, int(nodes[i, 0]) - 1))
        if j is not None:
            edges.append((j, i))
            n_cross += 1
    return dets, T, nodes, np.asarray(edges, int), n_cross


# ---- the sweep -----------------------------------------------------------
# One value changed at a time from the shipped configuration. Written down
# before the run; nothing is added afterwards.
# 0.08 dropped: thr3 showed it is below 0.12 on 6bba and flat against
# 0.10 on zebrahub, so it buys no shape and costs 45 minutes.
# TIGHTENING. Every earlier sweep went down; this is the untested
# direction, and the one whose local penalty and leaderboard penalty
# point opposite ways.
GATES = (0.15, 0.18, 0.22, 0.26)    # 0.15 is the shipped value
SHORTS = (4,)                       # what ships; thr4 showed additive
SWEEP = [(f"g{g:.2f}", g, sm) for sm in SHORTS for g in GATES]
BASE_NAME = f"g{GATES[0]:.2f}"
print(f"\n=== GATE SWEEP: {len(SWEEP)} thresholds x 3 domains ===",
      flush=True)
print("  PRE-REGISTERED RULE: a change counts only if its delta has the "
      "SAME SIGN in all three domains. One-domain wins are properties of "
      "that domain.", flush=True)
print("  BASE's 6bba row must reproduce biohub-poolsize's D187: score "
      "0.9386 / adj_J 0.9254 / div_J 0.1316 / TP-FP-FN 5-26-7.",
      flush=True)


# ---- STAGE B: the full pipeline, the only verdict that counts ----------
print(f"\n=== STAGE B: full pipeline on {len(HOLDOUT)} holdout datasets "
      f"(div_w {DIV_W}, global) ===", flush=True)
print("  BASE must reproduce biohub-div-gate's unpruned row: score 0.9363,"
      " adj_J 0.9220, div_J 0.1429, TP/FP/FN 5/23/7. If it does not, "
      "nothing else here is readable.", flush=True)

res = {}
def solve_one(dets, T):
    """Features, ranker, graph, ILP, and the repair pool. Reads the
    module-level UNET_THRESH through seeds_of and pair_features, so it
    must be re-run for every threshold -- unlike the post-link sweep,
    the gate changes the DETECTIONS."""
    seeds, vels, prev_of = pass1_tracks(dets)
    feats = pair_features(dets, None, T)
    probs = [MODEL.predict_proba(rows[:, MCOLS])[:, 1]
             if len(pairs) else np.zeros(0)
             for pairs, rows, P, C in feats]
    coords_rows, offset, n_glob = [], {}, 0
    for t in range(T):
        c, s_ = seeds[t]
        offset[t] = n_glob
        n_glob += len(c)
        for xyz in c:
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
    coords = np.asarray(np.round(coords_rows), np.int16)
    cand_edges = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= ILP_FLOOR:
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))
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
    nodes_arr = np.stack([np.asarray(na[c_].to_list(), float)
                          for c_ in ("t", "z", "y", "x")], axis=1)
    rows_e, dropped = [], 0
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for a_, b_ in zip(ea["source_id"].to_list(),
                          ea["target_id"].to_list()):
            ra, rb = remap.get(int(a_)), remap.get(int(b_))
            if ra is None or rb is None:
                dropped += 1
                continue
            rows_e.append([ra, rb])
    if dropped:
        warn_once("ilp_endpoint_missing",
                  f"{dropped} solved edges had an endpoint absent from "
                  f"node_attrs and were dropped -> a LOWER BOUND")
    e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
    nodes = {i_ + 1: {"t": int(nodes_arr[i_, 0]),
                      "xyz": nodes_arr[i_, 1:].astype(float)}
             for i_ in range(len(nodes_arr))}
    edges = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e2]
    cand, cand_sc, cand_trees = [], [], []
    n_rep = 0
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        n_rep += int(cm.sum())
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1])
                          else None)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    del graph, feats, probs
    return nodes, edges, n_glob, n_rep


def save_geff(nodes, edges, out_path):
    ids = sorted(nodes)
    rm = {g: i for i, g in enumerate(ids)}
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
    dr._save(str(out_path), arr, e3)


def pdir(cfg, domain):
    d = pathlib.Path(f"/kaggle/working/t3_{domain}_{cfg}")
    d.mkdir(parents=True, exist_ok=True)
    return d


res = {}
NODES = {}
REPAIR = {}
t_all = time.time()

for cfg, gate, short in SWEEP:
    UNET_THRESH = gate
    SHORT_MIN = short          # read by _short_filter as a module global
    tc = time.time()
    NODES[cfg] = 0
    REPAIR[cfg] = 0
    for domain, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
        for name in names:
            dets = load_dets(name)
            T = len(dets)
            nodes, edges, ng_, nr_ = solve_one(dets, T)
            NODES[cfg] += ng_
            REPAIR[cfg] += nr_
            save_geff(nodes, edges, pdir(cfg, domain) / f"{name}.geff")
            del dets, nodes, edges
            gc.collect()
        res[(cfg, domain)] = score_dir(pdir(cfg, domain))
    if SET_ZH:
        GT_DIR = pathlib.Path("/kaggle/working/t3_zh_gt")
        GT_DIR.mkdir(parents=True, exist_ok=True)
        zrows = []
        for name in SET_ZH:
            dets, T, gt_arr, gt_e, n_cross = zh_load(name)
            gp = GT_DIR / f"{name}.geff"
            dr._save(str(gp), gt_arr, gt_e)
            r_ = td.graph.IndexedRXGraph.from_geff(str(gp))
            gtg = r_[0] if isinstance(r_, tuple) else r_
            nodes, edges, ng_, nr_ = solve_one(dets, T)
            outp = pdir(cfg, "zh") / f"{name}.geff"
            save_geff(nodes, edges, outp)
            rr = td.graph.IndexedRXGraph.from_geff(str(outp))
            g = rr[0] if isinstance(rr, tuple) else rr
            er = compute_metric(g, gtg, scale=tuple(SCALE))
            rec = (node_recall(g, gtg)
                   if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
            zrows.append(per_sample_metrics(er, float(len(gt_arr)), rec))
            del dets, nodes, edges
            gc.collect()
        res[(cfg, "zh")] = _with_ratio(zrows)
    print(f"  {cfg}: nodes {NODES[cfg]}, repair pool {REPAIR[cfg]} "
          f"({time.time() - tc:.0f}s)", flush=True)

UNET_THRESH = GATES[0]
SHORT_MIN = SHORTS[0]

DOMAINS = ["6bba", "44b6"] + (["zh"] if SET_ZH else [])
BASE = BASE_NAME
print(f"\n=== absolute scores ===", flush=True)
print("  ANCHOR: gate 0.15 must reproduce thr4's g0.15_s4 row exactly -- 0.9279 / 0.8553 / 0.8037 -- same ranker (D187), same datasets, same SHORT_MIN 4. A drift means this assembly changed something it should not have.", flush=True)
print(f"  {'config':<10}" + "".join(f"{d + ' adj_J':>16}" for d in DOMAINS),
      flush=True)
for cfg, gate, short in SWEEP:
    print(f"  {cfg:<10}"
          + "".join(f"{res[(cfg, d)]['adj_edge_jaccard']:>16.4f}"
                    for d in DOMAINS), flush=True)

print(f"\n=== THE METRIC, DECOMPOSED ===", flush=True)
print("  adj_J = edge_J * (1 - 0.1 * node_ratio), and every sweep in "
      "this log has read only the product. edge_J says whether the "
      "EDGES got better; node_ratio is (N_pred - N_true)/N_true and says "
      "what the node count did. A NEGATIVE node_ratio means the "
      "pipeline emits FEWER nodes than there are cells, and the "
      "adjustment is then a BONUS rather than a penalty.", flush=True)
for d in DOMAINS:
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'gate':<8} {'edge_J':>9} {'node_ratio':>12} "
          f"{'adj_J':>9} {'div_J':>9}", flush=True)
    for cfg, gate, short in SWEEP:
        r = res[(cfg, d)]
        print(f"  {cfg:<8} {r['edge_jaccard']:>9.4f} "
              f"{r.get('total_node_ratio', float('nan')):>+12.4f} "
              f"{r['adj_edge_jaccard']:>9.4f} "
              f"{r['division_jaccard']:>9.4f}", flush=True)

print(f"\n=== adj_J DELTA vs thr 0.15, per domain ===", flush=True)
print(f"  {'config':<10}" + "".join(f"{d:>12}" for d in DOMAINS)
      + "   verdict", flush=True)
for cfg, gate, short in SWEEP:
    if cfg == BASE:
        continue
    ds = [res[(cfg, d)]["adj_edge_jaccard"]
          - res[(BASE, d)]["adj_edge_jaccard"] for d in DOMAINS]
    same = all(x > 0 for x in ds) or all(x < 0 for x in ds)
    if same and ds[0] > 0:
        verdict = "SAME SIGN, POSITIVE -- survives the rule"
    elif same:
        verdict = "same sign, negative"
    else:
        verdict = "SIGNS DISAGREE -- 6bba artefact, discard"
    print(f"  {cfg:<10}" + "".join(f"{x:>+12.4f}" for x in ds)
          + f"   {verdict}", flush=True)

print(f"\n=== div_J DELTA vs thr 0.15 ===", flush=True)
for cfg, gate, short in SWEEP:
    if cfg == BASE:
        continue
    ds = [res[(cfg, d)]["division_jaccard"]
          - res[(BASE, d)]["division_jaccard"] for d in DOMAINS]
    print(f"  {cfg:<10}" + "".join(f"{x:>+12.4f}" for x in ds), flush=True)

print("\n=== reading it ===", flush=True)
print("  biohub-thr-sweep measured +0.0033 adj_J at 0.12 on 6bba ALONE. "
      "If that sign does not hold on 44b6 and zebrahub, it belongs with "
      "biohub-poolsize: real on twelve same-batch datasets and absent "
      "everywhere else. The leaderboard has already refused two local "
      "gains of +0.0019 and +0.0073.", flush=True)
print("  The nodes and repair columns are printed because lowering the "
      "gate does two things at once -- it promotes detections into "
      "primary nodes AND shrinks _gap_close's repair pool by the same "
      "count. A fall could be either.", flush=True)
warning_summary()
print(f"\n  {time.time() - t_all:.0f}s total", flush=True)
print("\nTHR5 DONE", flush=True)