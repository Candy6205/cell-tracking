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

SCRIPT_VERSION = "2026-08-23e-ranker-v4b"
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
P_MIN_SWEEP = (0.30, 0.35, 0.40)           # A line
ILP_DIV_SWEEP = (0.75, 0.85, 0.95)         # B line
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

# ---------------- B line: capacity-2 ILP on ranker probs -----------------
print("\n--- B line: ILP(ranker probs), out-capacity 2 ---", flush=True)
for div_w in ILP_DIV_SWEEP:
    PRED = pathlib.Path(f"/kaggle/working/ilp_{int(div_w*100):02d}")
    PRED.mkdir(exist_ok=True)
    for name, (dets, feats, probs, T) in hold.items():
        # nodes: seeds at full res; candidate edges above the floor
        coords_rows = []
        offset = {}
        n_glob = 0
        for t in range(T):
            c, s = seeds_of(dets[t])
            offset[t] = n_glob
            n_glob += len(c)
            for xyz in c:
                coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
        coords = np.asarray(np.round(coords_rows), np.int16)
        cand_edges = []
        for t in range(T - 1):
            pairs, rows, Pp, Cc = feats[t]
            pr = probs[t]
            for (i, j), p in zip(pairs, pr):
                if p >= ILP_FLOOR:
                    cand_edges.append((offset[t] + i, offset[t + 1] + j,
                                       float(p), 0.0))
        graph = PIPE.build_graph(coords, cand_edges)
        if graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                appearance_weight=0.0,
                disappearance_weight=ILP_DISAPP,
                division_weight=div_w)
            with PIPE.suppress_output():
                graph = solver.solve(graph)
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        nid_l = [int(v) for v in na["node_id"].to_list()]
        remap = {v: i for i, v in enumerate(nid_l)}
        nodes_arr = np.stack([np.asarray(na[c].to_list(), float)
                              for c in ("t", "z", "y", "x")], axis=1)
        if graph.num_edges() > 0:
            ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
            # link_replay's warn_once("edge_endpoint_missing") case: the
            # solver can emit an edge whose endpoint is absent from
            # node_attrs. Dropping it loudly beats a KeyError that costs
            # the whole run (rule 1: degradation is never silent).
            rows, dropped = [], 0
            for a, b in zip(ea["source_id"].to_list(),
                            ea["target_id"].to_list()):
                ra, rb = remap.get(int(a)), remap.get(int(b))
                if ra is None or rb is None:
                    dropped += 1
                    continue
                rows.append([ra, rb])
            if dropped:
                print(f"  WARN {name} div_w={div_w:.2f}: dropped {dropped} "
                      f"ILP edges with absent endpoints -- this dataset's "
                      f"score is a LOWER BOUND", flush=True)
            e2 = np.array(rows, int) if rows else np.zeros((0, 2), int)
        else:
            e2 = np.zeros((0, 2), int)
        # patches on dict form
        nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                         "xyz": nodes_arr[i, 1:].astype(float)}
                 for i in range(len(nodes_arr))}
        edges = [(int(a) + 1, int(b) + 1) for a, b in e2]
        cand = []
        cand_sc = []
        cand_trees = []
        for t in range(T):
            c_all = np.asarray(dets[t][0], float)
            s_all = np.asarray(dets[t][1], float)
            cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
            cand.append(c_all[cm])
            cand_sc.append(s_all[cm])
            cand_trees.append(cKDTree(cand[-1] * SCALE)
                              if len(cand[-1]) else None)
        nid2 = len(nodes) + 1
        nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
        nodes, edges = _short_filter(nodes, edges)
        _linefit(nodes, edges)
        _add_safe_divisions(nodes, edges)
        ids = sorted(nodes)
        remap2 = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        e3 = np.array([[remap2[a], remap2[b]] for a, b in edges], int)
        dr._save(str(PRED / f"{name}.geff"), arr, e3)
    s = score_dir(PRED)
    print(f"div_w={div_w:.2f} ILP: score {s['score']:.4f}  "
          f"adj_J {s['adj_edge_jaccard']:.4f}  "
          f"div_J {s['division_jaccard']:.4f}  "
          f"(divTP {s['division_tp']} FP {s['division_fp']} "
          f"FN {s['division_fn']})", flush=True)

print("\nReferences: xl baseline 0.9318 (adj 0.9185 / div 0.1333); "
      "A-line v3 best 0.9320.", flush=True)
print("RANKER V4 DONE", flush=True)
