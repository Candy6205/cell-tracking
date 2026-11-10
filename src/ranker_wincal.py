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

SCRIPT_VERSION = "2026-08-23k-wincal"
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

# ---- retrain from caches (exclusion labels, same as v2) ----------------
train_names = sorted(n for n in CACHES if n not in HOLDOUT)
X_list, y_list = [], []
t0 = time.time()
for name in train_names:
    dets = load_dets(name)
    T = len(dets)
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
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
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
            X_list.append(row)
            y_list.append(1 if j in j_true else 0)

X = np.stack(X_list)
y = np.asarray(y_list)
print(f"training: {len(y)} pairs, {y.mean():.3f} pos, "
      f"{time.time()-t0:.0f}s", flush=True)
perm = np.random.default_rng(0).permutation(len(y))
n_val = max(1000, len(y) // 10)
vi, ti = perm[:n_val], perm[n_val:]
clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
clf.fit(X[ti], y[ti])
print(f"AUC: {roc_auc_score(y[vi], clf.predict_proba(X[vi])[:, 1]):.4f}",
      flush=True)
with open("/kaggle/working/edge_ranker.pkl", "wb") as fh:
    pickle.dump({"model": clf, "version": SCRIPT_VERSION}, fh)

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


# ---- holdout: features once, sweep p_min, full pipeline at each --------
hold = {}
for name in HOLDOUT:
    dets = load_dets(name)
    T = len(dets)
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
    probs = [clf.predict_proba(rows)[:, 1] if len(pairs) else np.zeros(0)
             for pairs, rows, P, C in feats]
    hold[name] = (dets, feats, probs, T)
print("holdout features ready", flush=True)

best = (None, -1)
for p_min in P_MIN_SWEEP:
    PRED = pathlib.Path(f"/kaggle/working/full_{int(p_min*100):02d}")
    PRED.mkdir(exist_ok=True)
    for name, (dets, feats, probs, T) in hold.items():
        nodes = {}
        frame_ids = []
        nid = 1
        cand = []
        cand_sc = []
        cand_trees = []
        for t in range(T):
            c_all, s_all = np.asarray(dets[t][0], float), \
                np.asarray(dets[t][1], float)
            seeds_m = s_all >= UNET_THRESH
            cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
            cand.append(c_all[cm])
            cand_sc.append(s_all[cm])
            cand_trees.append(cKDTree(cand[-1] * SCALE)
                              if len(cand[-1]) else None)
            ids = []
            for xyz in c_all[seeds_m]:
                nodes[nid] = {"t": t, "xyz": np.asarray(xyz, float)}
                ids.append(nid)
                nid += 1
            frame_ids.append(ids)
        edges = []
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            if not pairs:
                continue
            pr = probs[t]
            N, M = len(P), len(C)
            BIG = 1e9
            cost = np.full((N, M), BIG)
            for (i, j), p in zip(pairs, pr):
                if p >= p_min:
                    cost[i, j] = 1.0 - p
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] < BIG:
                    edges.append((frame_ids[t][int(a)],
                                  frame_ids[t + 1][int(b)]))
        nid = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid)
        nodes, edges = _short_filter(nodes, edges)
        _linefit(nodes, edges)
        _add_safe_divisions(nodes, edges)
        ids = sorted(nodes)
        remap = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        e2 = np.array([[remap[a], remap[b]] for a, b in edges], int)
        dr._save(str(PRED / f"{name}.geff"), arr, e2)
    s = score_dir(PRED)
    print(f"p_min={p_min:.2f} FULL: score {s['score']:.4f}  "
          f"adj_J {s['adj_edge_jaccard']:.4f}  "
          f"div_J {s['division_jaccard']:.4f}", flush=True)
    if s["score"] > best[1]:
        best = (p_min, s["score"])

print(f"\nA line best p_min {best[0]} -> {best[1]:.4f}  vs xl baseline "
      f"0.9318", flush=True)

# ---------------- windowed solve + calibrated probabilities -------------
# Two independent questions in one kernel:
#   W line -- does cutting the ILP into overlapping time windows keep the
#             score while buying the speed the 12 h cap needs?
#   C line -- the ranker's labels make BOTH GT children positive, which
#             inflates second-daughter probabilities; div_w 0.55 is partly
#             paying for that bias. Isotonic calibration should move the
#             optimum down and flatten the curve.

def _bucket_by_source(coords, cand_edges):
    tcol = coords[:, 0].astype(int)
    by_t = defaultdict(list)
    for e in cand_edges:
        by_t[int(tcol[e[0]])].append(e)
    return by_t


def solve_edges(coords, cand_edges, T, div_w, window_size=None):
    """Solve the ILP, optionally in time windows. Returns global edge pairs.

    window_size=None reproduces the single global solve that measured
    0.9363. Otherwise windows share ONE frame (ultrack's overlap_size=1)
    but never share an EDGE: window [a,b] owns a->a+1 .. (b-1)->b and the
    next window [b,c] owns b->b+1 .. So the merged edge set cannot double
    up, and no reconciliation pass is needed.
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
    empty_win = 0
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
            empty_win += 1
            continue
        graph = PIPE.build_graph(sub_coords, sub_edges)
        if graph.num_edges() == 0:
            empty_win += 1
            continue
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=div_w)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        # Map solved nodes back BY COORDINATE, never by node_id -- node_id
        # is not a stable handle across the solve (progress.md 02f7c8b).
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
            empty_win += 1
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
        print(f"    WARN {dropped} solved edges could not be matched back "
              f"by coordinate -- LOWER BOUND", flush=True)
    if empty_win:
        print(f"    WARN {empty_win} time windows produced no edges "
              f"-- those frames are UNLINKED, the movie is a LOWER BOUND",
              flush=True)
    return out


def run_config(tag, probs_by_name, div_w, window_size):
    """One full pipeline pass over the holdout. Returns (summary, seconds)."""
    PRED = pathlib.Path(f"/kaggle/working/cfg_{tag}")
    shutil.rmtree(PRED, ignore_errors=True)
    PRED.mkdir(parents=True)
    t_link = 0.0
    forks_ilp = forks_final = 0
    for name, (dets, feats, _p, T) in hold.items():
        probs = probs_by_name[name]
        t_a = time.time()
        coords_rows, offset, n_glob = [], {}, 0
        for t in range(T):
            c, s_ = seeds_of(dets[t])
            offset[t] = n_glob
            n_glob += len(c)
            for xyz in c:
                coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
        coords = np.asarray(np.round(coords_rows), np.int16)
        cand_edges = []
        for t in range(T - 1):
            pairs, rows, Pp, Cc = feats[t]
            if not pairs:
                continue
            pr = probs[t]
            for (i, j), p in zip(pairs, pr):
                if p >= ILP_FLOOR:
                    cand_edges.append((offset[t] + i, offset[t + 1] + j,
                                       float(p), 0.0))
        e_pairs = solve_edges(coords, cand_edges, T, div_w, window_size)
        t_link += time.time() - t_a
        oc = defaultdict(int)
        for a_, _b in e_pairs:
            oc[a_] += 1
        forks_ilp += sum(1 for v in oc.values() if v >= 2)
        nodes = {i + 1: {"t": int(coords[i, 0]),
                         "xyz": np.asarray(coords_rows[i][1:], float)}
                 for i in range(len(coords_rows))}
        edges = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e_pairs]
        cand, cand_sc, cand_trees = [], [], []
        for t in range(T):
            c_all = np.asarray(dets[t][0], float)
            s_all = np.asarray(dets[t][1], float)
            cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
            cand.append(c_all[cm])
            cand_sc.append(s_all[cm])
            cand_trees.append(cKDTree(cand[-1] * SCALE)
                              if len(cand[-1]) else None)
        _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)
        nodes, edges = _short_filter(nodes, edges)
        _linefit(nodes, edges)
        _add_safe_divisions(nodes, edges)
        oc2 = defaultdict(int)
        for a_, _b in edges:
            oc2[a_] += 1
        forks_final += sum(1 for v in oc2.values() if v >= 2)
        ids = sorted(nodes)
        rm = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        _keep = [[rm[a_], rm[b_]] for a_, b_ in edges
                 if a_ in rm and b_ in rm]
        if len(_keep) != len(edges):
            print(f"    WARN {len(edges) - len(_keep)} edges dropped: an "
                  f"endpoint survived no node -- this dataset scores as a "
                  f"LOWER BOUND", flush=True)
        e3 = np.array(_keep, int) if _keep else np.zeros((0, 2), int)
        dr._save(str(PRED / f"{name}.geff"), arr, e3)
    s = score_dir(PRED)
    print(f"{tag}: score {s['score']:.4f}  adj_J {s['adj_edge_jaccard']:.4f}"
          f"  div_J {s['division_jaccard']:.4f}  "
          f"(divTP {s['division_tp']} FP {s['division_fp']} "
          f"FN {s['division_fn']})  forks ilp {forks_ilp} final "
          f"{forks_final}  LINK {t_link:.0f}s", flush=True)
    return s, t_link


RAW = {n: hold[n][2] for n in hold}

# ---- C line needs calibrated probabilities -----------------------------
from sklearn.isotonic import IsotonicRegression                # noqa: E402
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(clf.predict_proba(X[vi])[:, 1], y[vi])
CAL = {n: [iso.predict(p) if len(p) else p for p in RAW[n]] for n in hold}
_r = np.concatenate([p for n in hold for p in RAW[n] if len(p)])
_c = np.concatenate([p for n in hold for p in CAL[n] if len(p)])
print(f"\ncalibration: raw mean {_r.mean():.4f} -> cal {_c.mean():.4f}; "
      f"frac>0.55 {(_r > 0.55).mean():.4f} -> {(_c > 0.55).mean():.4f}",
      flush=True)

print("\n--- W line: windowed ILP, uncalibrated, div_w 0.55 ---", flush=True)
w_rows = []
for ws in (None, 50, 25, 10):
    tag = f"w{ws if ws else 'global'}"
    s, tl = run_config(tag, RAW, 0.55, ws)
    w_rows.append((ws, s["score"], tl))

print("\n--- C line: calibrated probs, global solve, div_w sweep ---",
      flush=True)
c_rows = []
for dw in (0.05, 0.15, 0.30, 0.55):
    s, tl = run_config(f"c{int(dw*100):02d}", CAL, dw, None)
    c_rows.append((dw, s["score"], tl))

print("\n=== SUMMARY (xl baseline 0.9318; uncal global div_w 0.55 = "
      "0.9363) ===", flush=True)
print("W line (uncalibrated, div_w 0.55):", flush=True)
for ws, sc, tl in w_rows:
    base = w_rows[0][2]
    print(f"  window {str(ws):>6}  score {sc:.4f}  link {tl:6.0f}s  "
          f"speedup {base / max(tl, 1e-9):4.2f}x", flush=True)
print("C line (calibrated, global):", flush=True)
for dw, sc, tl in c_rows:
    print(f"  div_w {dw:.2f}  score {sc:.4f}  link {tl:6.0f}s", flush=True)
print("\nWINCAL DONE", flush=True)
