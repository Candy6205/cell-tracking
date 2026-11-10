"""fusescore -- the pipeline verdict on the fourth-heatmap blends.

SCRIPT_VERSION 2026-08-31a-mergescore

THE QUESTION. dcmerge REOPENED. 2026-08-28's dcmerge added the
PUBLISHED DeepCenter's points as primaries: zh +0.022..+0.038 monotone,
6bba negative monotone, 44b6 permanently nan (all 71 in that model's
training set) -- discarded on sign disagreement. Two things changed:
(1) the 6bba veto on detection-side changes is memorization-tainted
(F4 lost 6bba locally and won the LB +0.003, ref 55889500), so 6bba is
NON-VOTING here; (2) our ep05 excludes 12 44b6 sets from training, so
44b6 -- competition-style annotation, the domain that prices the
user's node-tax objection -- can finally vote. Base arm is F4 (the
shipped 0.925 detections); merge arms add ep05 points (score >= tau,
farther than MERGE_UM from every F4 point) as primaries at score
UNET_THRESH, dcmerge's admission rule verbatim. The user's framing on
record: zh's dense GT subsidises added points (no cell-count ceiling
hit), so the DECIDING vote is 44b6 -- added-point edge conversion vs
the linear node tax the LB charges (+10.3% output cost -0.002, twice
measured).

RANKER RULE (working-agreement, the leak): the arms differ in their
DETECTIONS, so the ranker's in-sample bias does not cancel; every
judged arm therefore runs D32, whose pool excludes the whole eval set
(the 44b6 twelve were drawn outside it by dctrain). One extra arm,
BASE x D187, exists ONLY to anchor the harness: its 6bba row must
reproduce dcpool's 0.9293. BASE x D32 is a new baseline and is the
in-run reference all deltas are taken against.

Solve runs PER ARM per dataset -- nothing upstream is shared when the
detections differ. Assembly: dcpool.py verbatim minus its sweep; the
driver replaced.

"""

from __future__ import annotations

import gc
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-31a-fatescore"
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
GAP1_CAP_FRAC = 0.003
# Shipped behaviour: bridge ONE missing frame, and only onto a real
# candidate from the repair band. Both are swept below; BASE must
# reproduce the recorded anchor with these values, which is what
# proves the rewritten _gap_close did not change behaviour.
GAP_SPAN = 1
GAP_NEED_CAND = True
# ---- the gates. Each is OFF at its neutral value, so BASE is unchanged.
# GAP_VEL_COS  minimum cosine between the incoming velocity at the gap's
#              start and the outgoing velocity at its end. -1.0 accepts
#              anything, which is the shipped behaviour.
# GAP_MIN_LEN  minimum node count of BOTH fragments being joined. 0 is off.
# GAP_CLEAR_UM a bridge is refused if a PRIMARY node already sits this
#              close to the interpolated position. 0.0 is off.
# GAP_LEN_MODE decides WHICH bridges GAP_MIN_LEN is allowed to judge.
#   "always"     every bridge. gap3's len4/len8 arms, and the reason
#                they delete bridges a real detected peak was backing.
#   "if_nocand"  only bridges with no candidate behind them. The user's
#                design: the candidate IS the evidence, the length
#                requirement is only the fallback for when there is none.
GAP_LEN_MODE = "always"
# ---- the second detector ------------------------------------------------
# None is OFF, and BASE is then byte-identical to cap20.
DC_TAU = None
DC_MERGE_UM = 4.0          # same separation our own NMS enforces
DC_ADD_SCORE = 0.12        # repair band only: >= MIN_CAND 0.10, < THRESH 0.15
_DC_STATS = [0, 0, 0]      # frames, xl pool points, DC points ADDED to pool
from collections import defaultdict as _dd
_SNAP_STATS = _dd(lambda: [0, 0, 0])   # DC_TAU -> [midpoint, xl snap, DC snap]
_CAND_IS_DC = None         # per-frame bool arrays, set by the arm loop
GAP_VEL_COS = -1.0
GAP_MIN_LEN = 0
GAP_CLEAR_UM = 0.0    # REBOUND per configuration
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


# dccache3's epoch-5 detections. The bounded walk prunes the competition
# and zarr/geff trees (trap 20).
def _walk_npz(prefix):
    import os
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


DCDET = {}          # unused in this kernel (no pool augmentation)

# fusecache's blended detections: fdet_<arm>_<name>.npz. Arm tags carry
# no underscore, so one split recovers (arm, name).
FDET = defaultdict(dict)
for _k, _p in _walk_npz("fdet_").items():
    _arm, _nm = _k.split("_", 1)
    FDET[_arm][_nm] = _p
print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)

# dccache3's epoch-5 detections, the ADD source for the merge arms.
DCDET_EP05 = _walk_npz("dcdet_ep05_")
print(f"dcdet_ep05 caches: {len(DCDET_EP05)}", flush=True)
MERGE_UM = 4.0    # dcmerge's DC_MERGE_UM: same separation our NMS enforces
_MERGE_STATS = defaultdict(lambda: [0, 0])   # (tau,dom) -> [f4 pts, added]


_ADDED_FRAMES = []      # per-frame count of admitted ep05 points
_FATE = {}              # one dataset's trace, refilled per solve


def _degree(edges):
    d = defaultdict(int)
    for a_, b_ in edges:
        d[a_] += 1
        d[b_] += 1
    return d


def merge_dets(dets, name, tau):
    """dcmerge's admission rule on an F4 base: an ep05 point enters iff
    its own score >= tau and it is farther than MERGE_UM from every F4
    point at that frame; it enters at score UNET_THRESH, i.e. as a
    PRIMARY node that never touches the repair band."""
    assert name in DCDET_EP05, f"dcdet_ep05 missing for {name}"
    dom_ = name.split("_")[0] if name.startswith(("6bba", "44b6")) \
        else "zh"
    st = _MERGE_STATS[(tau, dom_)]
    z = np.load(DCDET_EP05[name])
    out = []
    _ADDED_FRAMES.clear()
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        st[0] += len(c)
        dc = np.zeros((0, 3))
        if f"c{t}" in z.files:
            dc = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
            ds = np.asarray(z[f"s{t}"], float).reshape(-1)
            dc = dc[ds >= tau]
        if len(dc) and len(c):
            d, _ = cKDTree(c * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > MERGE_UM]
        st[1] += len(dc)
        _ADDED_FRAMES.append(len(dc))
        if len(dc):
            out.append((np.concatenate([c, dc], 0),
                        np.concatenate([s, np.full(len(dc),
                                                   UNET_THRESH)], 0)))
        else:
            out.append((c, s))
    return out


def _dc_pool_aug(pool, dets, name):
    """The REPAIR POOL with DeepCenter's epoch-5 peaks added, primaries
    untouched. Read by `_gap_close` only, so DC_TAU is a POST-LINK
    constant here (a constant belongs to the stage that reads it).

    A DC peak enters the pool when its own score is >= DC_TAU and it is
    further than DC_MERGE_UM from every PRIMARY node and every existing
    pool candidate at that frame -- add-only, no reordering, no joint
    NMS. Added points carry DC_ADD_SCORE (0.12): snap-eligible, never
    primary. Returns (cand, cand_sc, cand_trees, is_dc)."""
    cand, cand_sc, _trees = pool
    if DC_TAU is None or name not in DCDET:
        trees = [cKDTree(c * SCALE) if len(c) else None for c in cand]
        return (cand, cand_sc, trees,
                [np.zeros(len(c), bool) for c in cand])
    z = np.load(DCDET[name])
    c2, s2, is_dc = [], [], []
    for t in range(len(cand)):
        c = np.asarray(cand[t], float).reshape(-1, 3)
        sc = np.asarray(cand_sc[t], float).reshape(-1)
        kc = f"c{t}"
        prim, _ps = seeds_of(dets[t])
        _DC_STATS[0] += 1
        _DC_STATS[1] += len(c)
        dc = np.zeros((0, 3))
        if kc in z.files:
            dc = np.asarray(z[kc], float).reshape(-1, 3)
            ds = np.asarray(z[f"s{t}"], float).reshape(-1)
            dc = dc[ds >= DC_TAU]
        if len(dc) and len(prim):
            d, _ = cKDTree(prim * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > DC_MERGE_UM]
        if len(dc) and len(c):
            d, _ = cKDTree(c * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > DC_MERGE_UM]
        _DC_STATS[2] += len(dc)
        if len(dc):
            c2.append(np.concatenate([c, dc], 0))
            s2.append(np.concatenate([sc, np.full(len(dc), DC_ADD_SCORE)], 0))
            is_dc.append(np.concatenate([np.zeros(len(c), bool),
                                         np.ones(len(dc), bool)]))
        else:
            c2.append(c)
            s2.append(sc)
            is_dc.append(np.zeros(len(c), bool))
    trees = [cKDTree(c * SCALE) if len(c) else None for c in c2]
    return c2, s2, trees, is_dc


def load_dets(name, det="BASE"):
    """BASE reads the shipped detcache; F4 reads fusecache's fdet;
    MERGE:<tau> reads the F4 dets and adds ep05 points above tau."""
    if det.startswith("MERGE:"):
        return merge_dets(load_dets(name, "F4"), name,
                          float(det.split(":", 1)[1]))
    path = CACHES[name] if det == "BASE" else FDET[det][name]
    z = np.load(path)
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


_PRIM_TREES = {}

# made / allowed / demand-seen / times the cap bound, keyed by the arm's
# own constants so no tag has to be plumbed through the sweep loop.
# gap3 printed none of this, which is why it could not tell "len8 picks
# better bridges" apart from "len8 shrinks demand below the budget".
_GAP_LOG = defaultdict(lambda: [0, 0, 0, 0])


def _prim_tree(t, by_t, nodes):
    """KD-tree of the PRIMARY nodes at frame t, in micrometres. Cached
    per call of _gap_close -- the caller clears it."""
    if t in _PRIM_TREES:
        return _PRIM_TREES[t]
    ids = by_t.get(t, [])
    tr = None
    if ids:
        tr = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float)
                     * SCALE)
    _PRIM_TREES[t] = tr
    return tr


def _frag_len(nid, link, back, cap=12):
    """How many nodes the fragment containing `nid` runs for, walking
    backward through `pred` or forward through `succ`. Capped so a very
    long track costs nothing to measure."""
    n = 1
    cur = nid
    while n < cap:
        nxt = link.get(cur)
        if nxt is None:
            break
        cur = nxt
        n += 1
    return n


def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.

    Rewritten from the version that only ever bridged ONE missing frame
    (`ends` at t joined to `starts` at t+2). Two behaviours are now
    parameters, because flicker.py showed both matter and neither had
    ever been tested:

      GAP_SPAN        how many consecutive missing frames to bridge.
                      1 reproduces the shipped behaviour exactly.
      GAP_NEED_CAND   whether every inserted node must land on a real
                      candidate from the repair band. The shipped code
                      required it. At the missed frames 79.8% have NO
                      candidate within 5 um, so requiring it caps the
                      repair at about a fifth of dropouts, and a
                      two-frame bridge needing one at both frames would
                      fire on roughly 4%.

    With GAP_SPAN=1 and GAP_NEED_CAND=True this must be byte-identical
    in behaviour to the original -- that is the BASE arm's job to prove.
    """
    T = len(cand)
    _PRIM_TREES.clear()
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
    n_demand = 0
    BIG = 1e9
    # Shortest gaps first. A cell missing one frame should be repaired by
    # the confident one-frame rule before a longer, weaker bridge is
    # allowed to consume the same endpoint.
    for span in range(1, GAP_SPAN + 1):
        step = span + 1
        for t in range(T - step):
            if n_added >= max_gaps:
                break
            ends = [g for g in by_t.get(t, [])
                    if g not in succ and g in pred]
            starts = [g for g in by_t.get(t + step, [])
                      if g not in pred and g in succ]
            if not ends or not starts:
                continue
            P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
            C = np.asarray([nodes[g]["xyz"] for g in starts],
                           float) * SCALE
            D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2))
            # A longer bridge should be allowed proportionally more total
            # travel, otherwise a 2-frame bridge held to a 1-frame budget
            # would essentially never fire and the run would be testing
            # the gate rather than the span.
            #
            # ⚠ v1 WROTE `D = D / float(step)` AND THAT BROKE THE BASE
            # ARM. `step` is `span + 1`, so at span 1 -- the shipped
            # configuration -- it divided by 2 and silently doubled
            # GAP1_GATE_UM from 9 to 18 um. BASE then reproduced
            # biohub-poolsize's D187 row at adj_J 0.9253 / div_J 0.1282
            # against the recorded 0.9254 / 0.1316, which is exactly the
            # kind of small drift a must-reproduce anchor exists to
            # catch. The SHORT_MIN deltas survived it because the same
            # loosened gate sat under both halves and cancelled in the
            # difference; the GAP arms did not, because they compare
            # different spans and therefore different gates.
            #
            # Normalise by SPAN, not step: span 1 divides by 1 and is
            # byte-identical to the shipped code.
            D = D / float(span)
            cost = np.where(D > GAP1_GATE_UM, BIG, D)
            ri, ci = linear_sum_assignment(cost)
            for r_, c_ in zip(ri, ci):
                if n_added >= max_gaps or cost[r_, c_] >= BIG:
                    continue
                ge, gs = ends[r_], starts[c_]
                if ge in succ or gs in pred:
                    continue
                a_xyz = nodes[ge]["xyz"]
                b_xyz = nodes[gs]["xyz"]
                n_demand += 1

                # ---- the gates ---------------------------------------
                # Each is skipped at its neutral value so BASE is
                # byte-identical to the shipped behaviour.
                if GAP_VEL_COS > -1.0:
                    pa, sb = pred.get(ge), succ.get(gs)
                    if pa is None or sb is None:
                        continue
                    v_in = (a_xyz - nodes[pa]["xyz"]) * SCALE
                    v_out = (nodes[sb]["xyz"] - b_xyz) * SCALE
                    na, nb = np.linalg.norm(v_in), np.linalg.norm(v_out)
                    if na < 1e-6 or nb < 1e-6:
                        # a stationary endpoint carries no direction, so
                        # this gate cannot judge it. Let it through
                        # rather than silently rejecting every still cell.
                        pass
                    elif float(v_in @ v_out) / (na * nb) < GAP_VEL_COS:
                        continue
                if GAP_CLEAR_UM > 0.0:
                    blocked = False
                    for k in range(1, step):
                        frac = k / float(step)
                        m_ = ((1.0 - frac) * a_xyz + frac * b_xyz) * SCALE
                        tr_ = _prim_tree(t + k, by_t, nodes)
                        if tr_ is not None and \
                                tr_.query(m_)[0] <= GAP_CLEAR_UM:
                            blocked = True
                            break
                    if blocked:
                        continue

                # resolve every inserted position BEFORE mutating, so a
                # bridge that fails halfway leaves nothing behind
                picks = []
                for k in range(1, step):
                    frac = k / float(step)
                    mid = (1.0 - frac) * a_xyz + frac * b_xyz
                    tt = t + k
                    tree = cand_trees[tt]
                    hit = None
                    if tree is not None:
                        dist, idx = tree.query(mid * SCALE)
                        if dist <= GAP1_SNAP_UM \
                                and not cand_used[tt][idx] \
                                and cand_sc[tt][idx] >= GAP1_MIN_CAND_SCORE:
                            hit = idx
                    picks.append((tt, hit, mid))
                n_missing = sum(1 for _, _h, _ in picks if _h is None)

                # ---- the candidate requirement -----------------------
                if GAP_NEED_CAND and n_missing:
                    continue

                # ---- the length gate, moved to AFTER the lookup ------
                # gap3 ran this BEFORE resolving the candidate, which
                # cannot express "only when there is no candidate". The
                # lookup has no side effects (cand_used is marked in the
                # insert loop below, not above), so a bridge rejected
                # here leaves nothing behind and "always" reproduces
                # gap3 exactly. The three anchor arms test that.
                if GAP_MIN_LEN > 0 and (GAP_LEN_MODE == "always"
                                        or n_missing):
                    if _frag_len(ge, pred, back=True) < GAP_MIN_LEN or \
                            _frag_len(gs, succ, back=False) < GAP_MIN_LEN:
                        continue
                prev = ge
                for tt, hit, mid in picks:
                    _sn = _SNAP_STATS[DC_TAU]
                    if hit is None:
                        use = np.asarray(mid, float)
                        _sn[0] += 1
                    else:
                        use = np.asarray(cand[tt][hit], float)
                        cand_used[tt][hit] = True
                        if _CAND_IS_DC is not None \
                                and bool(_CAND_IS_DC[tt][hit]):
                            _sn[2] += 1
                        else:
                            _sn[1] += 1
                    ng = next_id
                    next_id += 1
                    nodes[ng] = {"t": tt, "xyz": use}
                    edges.append((prev, ng))
                    succ[prev] = ng
                    pred[ng] = prev
                    prev = ng
                edges.append((prev, gs))
                succ[prev] = gs
                pred[gs] = prev
                n_added += 1
    _r = _GAP_LOG[(GAP_NEED_CAND, GAP_SPAN, round(GAP1_GATE_UM, 2),
                   round(GAP1_CAP_FRAC, 5))]
    _r[0] += n_added
    _r[1] += max_gaps
    _r[2] += n_demand
    # ⚠ n_demand stops accumulating once the cap binds, because the
    # frame loop breaks there. So demand-seen UNDERSTATES true demand
    # for any capped arm -- read the cap20 arms for the real figure.
    _r[3] += 1 if n_added >= max_gaps else 0
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


def _adj_of(row):
    """adj_edge_jaccard for ONE dataset row.

    `summarise` builds that column; nothing on record says
    `per_sample_metrics` also puts it on the row. Use the key when it is
    there and reconstruct it from the definition when it is not, rather
    than letting a KeyError kill a print block that runs after four
    hours of compute.
    """
    v = row.get("adj_edge_jaccard")
    if v is not None and v == v:
        return float(v)
    j = float(row.get("edge_jaccard", float("nan")))
    r = float(row.get("total_node_ratio", 0.0) or 0.0)
    return max(0.0, j * (1.0 - 0.1 * r))


def _with_ratio(rows):
    """summarise(), plus the node-count column it does not aggregate.

    adj_J = edge_J * (1 - 0.1 * total_node_ratio), and `summarise` drops
    the middle term, so a sweep reading only adj_J cannot say whether a
    change improved the EDGES or merely the NODE COUNT. Weighted by
    tp+fp+fn, the same weights adj_edge_jaccard uses.

    Purely ADDITIVE: summarise's own keys are untouched. Lifted from
    thr5, where it was unit-tested.
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
    per = {}
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
        row = per_sample_metrics(er, float(est) if est else
                                 float("nan"), rec)
        rows.append(row)
        per[name] = row
    # per-dataset rows come back too: a weighted mean can hide one
    # dataset carrying an entire delta, which is exactly what this run
    # exists to check.
    return _with_ratio(rows), per


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


# ---- BOTH rankers, loaded not retrained --------------------------------
# The rule (working-agreement, ranker leak): a sweep that CHANGES THE
# DETECTIONS must use D32, because each arm feeds the ranker different
# candidates and D187's in-sample bias does not cancel in the deltas.
# Our 44b6 eval set is outside D32's pool by construction (dctrain drew
# it there), so D32 keeps all three domains clean. D187 is loaded too,
# for ONE anchor arm: BASE x D187 on 6bba must reproduce 0.9293, which
# proves the harness against dcpool before any new number is read.
def _load_ranker(fname, min_pool, max_pool):
    pk = None
    import os as _os
    for _root, _dirs, _fns in _os.walk("/kaggle/input"):
        _dirs[:] = [d for d in _dirs
                    if "cell-tracking-during-development" not in d
                    and not d.endswith(".zarr") and not d.endswith(".geff")]
        if fname in _fns:
            pk = pathlib.Path(_root) / fname
            break
    assert pk is not None, (f"{fname} not found -- attach "
                            f"faustine649/biohub-poolsize as kernel source")
    with open(pk, "rb") as f:
        blob = pickle.load(f)
    pool = blob.get("pool") if isinstance(blob, dict) else None
    assert pool is not None and min_pool <= len(pool) <= max_pool, (
        f"{pk}: pool {0 if pool is None else len(pool)} outside "
        f"[{min_pool}, {max_pool}]")
    print(f"  ranker {fname}: pool {len(pool)}", flush=True)
    return blob["model"], blob["cols"], set(pool)


MODEL_D187, MCOLS_D187, _POOL187 = _load_ranker(
    "edge_ranker_d187.pkl", 100, 199)
MODEL_D32, MCOLS_D32, _POOL32 = _load_ranker(
    "edge_ranker_d32.pkl", 1, 99)
MODEL, MCOLS = MODEL_D32, MCOLS_D32     # rebound per arm in the driver


def set_ranker(tag):
    global MODEL, MCOLS
    if tag == "D187":
        MODEL, MCOLS = MODEL_D187, MCOLS_D187
    else:
        MODEL, MCOLS = MODEL_D32, MCOLS_D32


# ---- the three evaluation domains --------------------------------------
# 6bba is poolsize's holdout verbatim. 44b6 is chosen AT RUNTIME from
# whatever caches are mounted, sorted, so no dataset name is invented
# here and a missing mount fails loudly instead of silently shrinking the
# comparison.
#
# ⚠ 2026-08-27, THE USER'S STANDING INSTRUCTION: "所有测试集的 44b6 都要
# 随机抽". Every sweep before this one used `sorted(...)[:12]` -- the
# twelve LOWEST hex ids out of 71, identical in every run, and nobody
# ever checked whether those ids encode acquisition order, embryo or
# session. An arbitrary-looking sort key is not a random sample, and a
# fixed prefix of it is the worst case: if the ids carry any structure,
# every three-domain verdict this project has issued shares one skew.
#
# Random, but SEEDED, so the sample is reproducible and a recorded
# anchor still means something. The chosen names are PRINTED every run
# -- a sample nobody can see is a sample nobody can check.
#
# To test whether a result depends on the sample, change SEED_44B6 and
# re-run; that is a different question from changing N_44B6, which
# changes the precision.
# The 44b6 arm is dctrain's EXCLUDED 12, read from the training run's
# own split_manifest.json rather than recomputed -- if the training
# split ever changes, this fails loudly instead of silently evaluating
# on movies the model saw. These 12 were drawn seeded (20260827) from
# the 44b6 outside D32's pool, satisfying both the user's randomness
# rule and the ranker-leak rule. The epoch-5 model never saw them, so
# **this is the first 44b6 arm in the DeepCenter line that means
# anything, and the first time this line can face the three-domain
# rule.**
import json as _json                                           # noqa: E402
_MAN = None
import os as _os                                               # noqa: E402
for _root, _dirs, _fns in _os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs
                if "cell-tracking-during-development" not in d
                and not d.endswith(".zarr") and not d.endswith(".geff")]
    if "split_manifest.json" in _fns:
        _MAN = pathlib.Path(_root) / "split_manifest.json"
        break
assert _MAN is not None, ("split_manifest.json not found -- attach "
                          "faustine649/biohub-dctrain-175-weights")
_seen = set(_json.loads(_MAN.read_text())["all"])
SET_6BBA = [n for n in HOLDOUT if n in CACHES]
SET_44B6 = sorted(n for n in CACHES
                  if n.startswith("44b6_") and n not in _seen)
print(f"  44b6: {len(SET_44B6)} = dctrain's excluded set, from {_MAN}",
      flush=True)
print(f"    {' '.join(SET_44B6)}", flush=True)
assert len(SET_6BBA) == len(HOLDOUT), (
    f"only {len(SET_6BBA)} of {len(HOLDOUT)} holdout caches are mounted; "
    f"the 6bba row would not be comparable to the recorded anchor")
assert len(SET_44B6) == 12, (
    f"expected 12 excluded 44b6, found {len(SET_44B6)} -- the manifest "
    f"and the caches disagree")
assert all(n not in _seen for n in SET_6BBA), (
    "a 6bba holdout dataset appears in dctrain's training pool")
print(f"  domains: 6bba {len(SET_6BBA)}, 44b6 {len(SET_44B6)}", flush=True)

ZH = _walk_npz("zhdet_")
N_ZH = 8
SET_ZH = sorted(ZH)[:N_ZH]
print("  zebrahub is BACK. wide44 dropped it because its question was entirely about the 44b6 sample; this run asks whether a wider bridge pays, which is a candidate change and therefore a three-domain question.", flush=True)
# The inherited WARN here said "biohub-zh-detect is not attached". That
# would be a FALSE diagnosis in this run -- the mount is fine and the
# domain is dropped by choice. Report the reason that is true.
if not SET_ZH:
    print("  WARN biohub-zh-detect is not attached -> the zebrahub domain "
          "is ABSENT and the three-domain rule degrades to two. Say so in "
          "the write-up rather than reporting a two-domain result as if "
          "it satisfied the rule.", flush=True)
else:
    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)
    _mzh = [n for n in SET_ZH
            if n not in FDET.get("F4", {}) or n not in DCDET_EP05]
    assert not _mzh, f"F4/dcdet_ep05 missing for zh {_mzh}"


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
# ---- CROSSED, at the user's instruction --------------------------------
# v1 swept the gap arms with SHORT_MIN fixed at 6. Two things were wrong
# with that:
#
# 1. **6 is no longer the baseline.** biohub-short-sweep found 4 better
#    in all three domains and it has been submitted. Measuring gap
#    changes against 6 measures them against a configuration we are
#    abandoning.
# 2. **The two knobs are the same phenomenon from opposite ends**, so
#    they cannot be assumed independent. flicker.py: a dropout longer
#    than one frame shatters a track, `_gap_close` reaches only one
#    frame, and `_short_filter` then deletes the fragments. If a wider
#    bridge repairs those dropouts there are fewer fragments to delete,
#    and SHORT_MIN's optimum MOVES. One-at-a-time sweeping assumes no
#    interaction exactly where a mechanism predicts one.
#
# So every gap arm runs at BOTH SHORT_MIN values. Two anchors fall out
# of that for free, and both must hold or nothing here is readable:
#   sm6 BASE  -> biohub-poolsize's D187 row, 6bba adj_J 0.9254
#   sm4 BASE  -> biohub-short-sweep's short_4: +0.0025 / +0.0054 / +0.0045
#
# Written as full dicts rather than (key, value): the combination arms
# move three constants at once and the one-key form cannot say that.
# All arms run at SHORT_MIN 4, the shipped value. gap2 already showed
# the gap arms and SHORT_MIN interact, so holding it at what ships is
# the only setting whose answer is actionable.
# (The dcmerge/dcpool sweep machinery that lived here is gone: this
# kernel has no config sweep. RUN_ARMS in the driver below carries
# the arms -- detections x ranker -- and the shipped constants are
# bound once by apply_config there.)
SWEEP = []
COMBOS = []

res = {}
PER = {}   # (cfg, domain) -> {dataset: row}
BASE_VALUES = {k: globals()[k] for k in
               ("GAP1_GATE_UM", "GAP1_SNAP_UM", "GAP1_MIN_CAND_SCORE",
                "GAP1_CAP_FRAC", "SHORT_MIN", "LINEFIT_WEIGHT",
                "LINEFIT_WINDOW", "GAP_SPAN", "GAP_NEED_CAND",
                "GAP_VEL_COS", "GAP_MIN_LEN", "GAP_CLEAR_UM",
                "GAP_LEN_MODE", "DC_TAU")}


# ⚠ In dcmerge DC_TAU sat in this set because `_dc_merge` ran inside
# `load_dets`. In THIS file the DC augmentation moved to the pool stage
# (`_dc_pool_aug`, called inside the arm loop), so DC_TAU is a POST-LINK
# constant here and is deliberately NOT in the set. The rule that
# decides membership: find the line that READS the constant and check
# whether it runs inside or outside the arm loop.
DETECTION_TIME_CONSTANTS = {"UNET_THRESH", "CAND_THR", "NMS_UM"}


def apply_config(key, val):
    """Rebind swept constants, or restore all of them for BASE.

    A CONSTANT BELONGS TO THE STAGE THAT READS IT. Everything this
    sweep varies (DC_TAU included -- see the note above) is read by the
    post-link stages, which run inside the arm loop; `load_dets` and
    `solve_one` read none of it, so the solve is shared across arms.

    The patch functions read these as module globals, so this is what
    "changing one thing" means here.

    `key` may be a str (one constant) or a DICT (several at once).
    """
    for k, v in BASE_VALUES.items():
        globals()[k] = v
    if isinstance(key, dict):
        for k, v in key.items():
            assert k in BASE_VALUES, f"{k} is not a swept constant"
            globals()[k] = v
    elif key is not None:
        assert key in BASE_VALUES, f"{key} is not a swept constant"
        globals()[key] = val


def solve_one(dets, T):
    """Everything up to and including the ILP: features, ranker, graph,
    solve, and the sub-threshold repair pool. NONE of it depends on a
    constant being swept, so it runs ONCE per dataset and every
    configuration is applied to a copy of its output."""
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
    nodes0 = {i_ + 1: {"t": int(nodes_arr[i_, 0]),
                       "xyz": nodes_arr[i_, 1:].astype(float)}
              for i_ in range(len(nodes_arr))}
    edges0 = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e2]
    cand, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1])
                          else None)
    # --- fate trace, part 1: everything up to and including the ILP.
    # Work in (t, i) SEED space -- that is the space `feats` and `probs`
    # already speak -- and cross to graph ids only by rounded
    # coordinates, edgediag's technique. Nothing here assumes the graph
    # numbers its nodes in coords order.
    _id_of = {}
    for _g, _v in nodes0.items():
        _id_of[(int(_v["t"]), tuple(int(round(float(_x)))
                                    for _x in _v["xyz"]))] = _g
    _seed_id, _unmapped = {}, 0
    for _t in range(T):
        _P = seeds[_t][0]
        for _i in range(len(_P)):
            _g = _id_of.get((_t, tuple(int(round(float(_x)))
                                       for _x in _P[_i])))
            if _g is None:
                _unmapped += 1
            else:
                _seed_id[(_t, _i)] = _g
    # Merged points are APPENDED by merge_dets and enter at exactly
    # UNET_THRESH, so seeds_of's `>= UNET_THRESH` mask keeps every one
    # and keeps them last: the tail k_t of frame t are the ep05 adds.
    _nseed = [len(seeds[_t][0]) for _t in range(T)]
    _merged = set()
    for _t in range(T):
        _k = _ADDED_FRAMES[_t] if _t < len(_ADDED_FRAMES) else 0
        for _m in range(_k):
            _merged.add((_t, _nseed[_t] - _k + _m))
    _in_pair, _past_floor = set(), set()
    for _t in range(T - 1):
        _pairs = feats[_t][0]
        for (_i, _j), _pr in zip(_pairs, probs[_t]):
            _in_pair.add((_t, _i))
            _in_pair.add((_t + 1, _j))
            if _pr >= ILP_FLOOR:
                _past_floor.add((_t, _i))
                _past_floor.add((_t + 1, _j))
    _key2ti = {_g: _ti for _ti, _g in _seed_id.items()}
    _deg0 = _degree(edges0)
    _ilp_linked = {_key2ti[_g] for _g in _deg0 if _g in _key2ti}
    _FATE.clear()
    _FATE.update({"merged": _merged, "nseed": _nseed, "T": T,
                  "in_pair": _in_pair, "past_floor": _past_floor,
                  "ilp_linked": _ilp_linked, "seed_id": _seed_id,
                  "key2ti": _key2ti, "unmapped": _unmapped,
                  "ilp_nodes0": set(nodes0)})
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)


def patch_and_save(nodes0, edges0, pool, out_path):
    """The four post-link stages on a FRESH COPY, then write the geff."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    _d_ilp = _degree(edges)
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    _d_gap = _degree(edges)
    _pre_short = set(nodes)
    nodes, edges = _short_filter(nodes, edges)
    _post_short = set(nodes)
    _d_short = _degree(edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    _FATE.update({"d_ilp": _d_ilp, "d_gap": _d_gap, "d_short": _d_short,
                  "d_final": _degree(edges),
                  "killed_short": _pre_short - _post_short,
                  "post_short": _post_short,
                  "final_nodes": set(nodes)})
    ids = sorted(nodes)
    rm = {g: i for i, g in enumerate(ids)}
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
    dr._save(str(out_path), arr, e3)


def pdir(cfg, domain):
    d = pathlib.Path(f"/kaggle/working/ps_{domain}_{cfg}")
    d.mkdir(parents=True, exist_ok=True)
    return d


res = {}
PER = {}
t_all = time.time()

# (arm, domain) -> group -> counter name -> int
FATE = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# (arm, domain) -> group -> final degree -> count
DEGHIST = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# (name, arm) -> frozenset of (t, i) BASE seeds the ILP linked
BASE_LINKED = {}
UNMAPPED = defaultdict(int)


def collect_fate(lbl, domain, name):
    """Fold one solved dataset's trace into the running totals.

    Three groups, kept apart because they are three different
    populations: BASE (F4's own detections), MERGED (the ep05 adds), and
    REPAIR (nodes _gap_close created from the sub-threshold pool, which
    exist in neither seed set)."""
    F = FATE[(lbl, domain)]
    H = DEGHIST[(lbl, domain)]
    merged, seed_id = _FATE["merged"], _FATE["seed_id"]
    killed, final = _FATE["killed_short"], _FATE["final_nodes"]
    d_ilp, d_gap = _FATE["d_ilp"], _FATE["d_gap"]
    d_short, d_fin = _FATE["d_short"], _FATE["d_final"]
    UNMAPPED[(lbl, domain)] += _FATE["unmapped"]
    base_linked = set()
    for t_ in range(_FATE["T"]):
        for i_ in range(_FATE["nseed"][t_]):
            ti = (t_, i_)
            grp = "merged" if ti in merged else "base"
            r = F[grp]
            r["primary"] += 1
            r["in_pair"] += ti in _FATE["in_pair"]
            r["past_floor"] += ti in _FATE["past_floor"]
            lk = ti in _FATE["ilp_linked"]
            r["ilp_linked"] += lk
            if grp == "base" and lk:
                base_linked.add(ti)
            g = seed_id.get(ti)
            if g is None:
                r["absent_from_graph"] += 1
                continue
            r["gap_gain"] += d_gap.get(g, 0) > d_ilp.get(g, 0)
            r["short_kill"] += g in killed
            r["div_gain"] += d_fin.get(g, 0) > d_short.get(g, 0)
            if g in final:
                r["final"] += 1
                deg = d_fin.get(g, 0)
                r["final_deg_sum"] += deg
                r["final_isolated"] += deg == 0
                H[grp][min(deg, 3)] += 1
    # REPAIR nodes: in the final graph but in no seed set.
    seed_keys = set(seed_id.values())
    for g in final:
        if g not in seed_keys:
            r = F["repair"]
            r["primary"] += 1
            r["final"] += 1
            deg = d_fin.get(g, 0)
            r["final_deg_sum"] += deg
            r["final_isolated"] += deg == 0
            H["repair"][min(deg, 3)] += 1
    BASE_LINKED[(name, lbl)] = frozenset(base_linked)

# The shipped post-link configuration, set ONCE (no sweep in this
# kernel): cap20 + no candidate requirement + SHORT_MIN 4, ref
# 55817630's. Every arm shares it; the arms differ in DETECTIONS and
# RANKER only.
apply_config({"GAP_NEED_CAND": False, "GAP1_CAP_FRAC": 0.02,
              "SHORT_MIN": 4}, None)

# (label, detection set, ranker)
RUN_ARMS = [
    ("F4",  "F4",         "D32"),    # control: the same points, no adds
    ("M15", "MERGE:0.15", "D32"),    # the arm whose points we are tracing
]
# mergescore's recorded rows. If these do not reproduce, the trace
# describes a different run than the one it is being used to explain.
ANCHOR_ADJ = {("F4", "6bba"): 0.9198, ("F4", "44b6"): 0.8303,
              ("F4", "zh"): 0.8115,
              ("M15", "6bba"): 0.9134, ("M15", "44b6"): 0.8363,
              ("M15", "zh"): 0.8219}
_missing = [n for n in SET_6BBA + SET_44B6
            if n not in FDET.get("F4", {}) or n not in DCDET_EP05]
assert not _missing, (
    f"F4/dcdet_ep05 cache missing for {_missing[:4]} -- attach "
    f"biohub-fusecache AND biohub-dccache3")
assert not (set(SET_44B6) & _POOL32), (
    "a 44b6 eval dataset sits in D32's pool -- the leak rule would be "
    "violated; dctrain's draw should have made this impossible")

print(f"\n=== MERGESCORE: {len(RUN_ARMS)} arms x 3 domains, one solve per "
      f"ARM per dataset (the detections differ, so nothing is shared) "
      f"===", flush=True)
print("  PRE-REGISTERED RULE (amended 2026-08-31): a merge arm counts only "
      "if its adj_J delta vs F4 (both on D32) has the SAME SIGN in 44b6 "
      "AND zh. 6bba is reported but NON-VOTING: memorization-tainted "
      "for detection-side changes (F4 lost 6bba locally, won the LB).", flush=True)
print("  RANKER: D32 for every judged arm -- a detection change feeds "
      "the ranker different candidates, so D187 in-sample bias would "
      "not cancel; the 44b6 eval set is outside the D32 pool by "
      "construction. BASE_D187 runs only as the harness anchor (6bba "
      "must read adj_J 0.9293).", flush=True)

for domain, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    td0 = time.time()
    for name in names:
        for lbl, det, rk in RUN_ARMS:
            set_ranker(rk)
            _ADDED_FRAMES.clear()
            dets = load_dets(name, det)
            T = len(dets)
            nodes0, edges0, pool = solve_one(dets, T)
            patch_and_save(nodes0, edges0, pool,
                           pdir(lbl, domain) / f"{name}.geff")
            collect_fate(lbl, domain, name)
            del dets, nodes0, edges0, pool
            gc.collect()
    for lbl, _d, _r in RUN_ARMS:
        res[(lbl, domain)], PER[(lbl, domain)] = score_dir(pdir(lbl, domain))
    print(f"  {domain}: {len(names)} datasets x {len(RUN_ARMS)} arms "
          f"({time.time() - td0:.0f}s)", flush=True)

if SET_ZH:
    tz = time.time()
    GT_DIR = pathlib.Path("/kaggle/working/ps_zh_gt")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    zrows = {lbl: [] for lbl, _d, _r in RUN_ARMS}
    for name in SET_ZH:
        dets_base, T, gt_arr, gt_e, n_cross = zh_load(name)
        gp = GT_DIR / f"{name}.geff"
        dr._save(str(gp), gt_arr, gt_e)
        r_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        gtg = r_[0] if isinstance(r_, tuple) else r_
        for lbl, det, rk in RUN_ARMS:
            set_ranker(rk)
            _ADDED_FRAMES.clear()
            if det == "BASE":
                dets = dets_base
            else:
                z_ = np.load(FDET["F4"][name])
                dets = [(z_[f"c{t}"], z_[f"s{t}"]) for t in range(T)]
                if det.startswith("MERGE:"):
                    dets = merge_dets(dets, name,
                                      float(det.split(":", 1)[1]))
            nodes0, edges0, pool = solve_one(dets, T)
            outp = pdir(lbl, "zh") / f"{name}.geff"
            patch_and_save(nodes0, edges0, pool, outp)
            collect_fate(lbl, "zh", name)
            rr = td.graph.IndexedRXGraph.from_geff(str(outp))
            g = rr[0] if isinstance(rr, tuple) else rr
            er = compute_metric(g, gtg, scale=tuple(SCALE))
            rec = (node_recall(g, gtg)
                   if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
            zrows[lbl].append(per_sample_metrics(er, float(len(gt_arr)),
                                                 rec))
            del dets, nodes0, edges0, pool
            gc.collect()
        del dets_base
        gc.collect()
    for lbl, _d, _r in RUN_ARMS:
        res[(lbl, "zh")] = summarise(zrows[lbl])
    print(f"  zebrahub: {len(SET_ZH)} crops x {len(RUN_ARMS)} arms "
          f"({time.time() - tz:.0f}s)", flush=True)

# ---- report -------------------------------------------------------------
DOMAINS = ["6bba", "44b6"] + (["zh"] if SET_ZH else [])
print("\n=== ANCHOR ===", flush=True)
print("  BASE_D187 6bba must read adj_J 0.9293 (dcpool). F4 x D32 is "
      "the in-run reference the merge arms are judged against; its own "
      "recorded row (fusescore) is 6bba 0.9198 / 44b6 0.8303 / zh "
      "0.8115 and must reproduce.", flush=True)
for lbl in ("BASE_D187", "F4"):
    for d in DOMAINS:
        r = res.get((lbl, d))
        if r:
            print(f"  {lbl:<10} {d:<5} score {r['score']:.4f}  adj_J "
                  f"{r['adj_edge_jaccard']:.4f}  edge_J "
                  f"{r.get('edge_jaccard', float('nan')):.4f}  div_J "
                  f"{r['division_jaccard']:.4f}", flush=True)

print("\n=== adj_J DELTA vs F4 (both D32), per domain ===", flush=True)
hdr = "  " + "arm".ljust(10)
for d in DOMAINS:
    hdr += d.rjust(12)
print(hdr + "   verdict", flush=True)
for lbl, det, rk in RUN_ARMS:
    if lbl in ("F4", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["adj_edge_jaccard"]
          - res[("F4", d)]["adj_edge_jaccard"] for d in DOMAINS]
    vote = ds[1:]     # 44b6 and zh; 6bba (ds[0]) reported, non-voting
    same = all(x > 0 for x in vote) or all(x < 0 for x in vote)
    verdict = ("CLEAN DOMAINS POSITIVE" if same and vote[0] > 0 else
               "clean domains negative" if same else
               "clean domains disagree -- discard")
    row = "  " + lbl.ljust(10)
    for x in ds:
        row += f"{x:>+12.4f}"
    print(row + "   " + verdict, flush=True)

print("\n=== div_J DELTA vs F4 (D32) ===", flush=True)
for lbl, det, rk in RUN_ARMS:
    if lbl in ("F4", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["division_jaccard"]
          - res[("F4", d)]["division_jaccard"] for d in DOMAINS]
    row = "  " + lbl.ljust(10)
    for x in ds:
        row += f"{x:>+12.4f}"
    print(row, flush=True)

print("\n=== decomposition ===", flush=True)
for d in DOMAINS:
    print(f"\n  -- {d} --", flush=True)
    print("  " + "arm".ljust(10) + "edge_J".rjust(9)
          + "node_ratio".rjust(12) + "adj_J".rjust(9) + "div_J".rjust(9),
          flush=True)
    for lbl, _det, _rk in RUN_ARMS:
        r = res.get((lbl, d))
        if r:
            print(f"  {lbl:<10} {r.get('edge_jaccard', float('nan')):>9.4f} "
                  f"{r.get('total_node_ratio', float('nan')):>+12.4f} "
                  f"{r['adj_edge_jaccard']:>9.4f} "
                  f"{r['division_jaccard']:>9.4f}", flush=True)

print("\n=== per-dataset, F4 vs M30 (both D32) ===", flush=True)
for d in DOMAINS:
    pb, pn = PER.get(("F4", d)), PER.get(("M30", d))
    if not pb or not pn:
        continue
    print(f"\n  -- {d} --", flush=True)
    for nm in sorted(pb):
        if nm not in pn:
            continue
        a, b = pb[nm], pn[nm]
        aa, ba = _adj_of(a), _adj_of(b)
        print(f"  {nm:<20} {aa:>9.4f} {ba:>9.4f} {ba - aa:>+9.4f}",
              flush=True)

print("\n=== points added, per tau x domain ===", flush=True)
for (tau_, dom_) in sorted(_MERGE_STATS):
    p_, a_ = _MERGE_STATS[(tau_, dom_)]
    print(f"  tau {tau_:.2f} {dom_:<5} +{a_:>8,} on {p_:>9,} F4 points "
          f"(+{a_ / max(1, p_):.2%} nodes)", flush=True)

def _pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "  -  "


print("\n=== ANCHORS: mergescore's rows must reproduce ===", flush=True)
_bad = 0
for (_l, _d), _want in sorted(ANCHOR_ADJ.items()):
    _got = res.get((_l, _d), {}).get("adj_edge_jaccard", float("nan"))
    _ok = abs(_got - _want) < 0.0002
    _bad += not _ok
    print(f"  {_l:<4} {_d:<5} adj_J {_got:.4f} (mergescore {_want:.4f}) "
          f"{'OK' if _ok else 'MISMATCH'}", flush=True)
print("  " + ("all anchors hold" if not _bad else
      f"{_bad} MISMATCHED -- the trace describes a different run"),
      flush=True)
_um = sum(UNMAPPED.values())
print(f"  seeds that never reached the solved graph: {_um:,} "
      f"(these are counted in `primary` and in `absent_from_graph`, "
      f"never in the post-link columns)", flush=True)

print("\n=== EVERY STAGE, SEPARATELY, BY GROUP ===", flush=True)
print("  base   = F4's own detections. merged = the ep05 points M15 "
      "adds. repair = nodes _gap_close created from the sub-threshold "
      "pool (they are in no seed set, so only their output columns "
      "exist).", flush=True)
print("  Percentages are of that group's OWN primary count, so base and "
      "merged rows are directly comparable.", flush=True)
for _d in ("6bba", "44b6", "zh"):
    for _l, _det, _rk in RUN_ARMS:
        if (_l, _d) not in FATE:
            continue
        print(f"\n  -- {_l} / {_d} --", flush=True)
        print(f"  {'group':<7} {'primary':>9} {'in_graph':>9} "
              f"{'in_pair':>8} {'floor':>8} {'ILP':>8} {'gap+':>7} "
              f"{'short-':>7} {'div+':>6} {'final':>8} {'ISOLATED':>9} "
              f"{'deg':>6}", flush=True)
        for _g in ("base", "merged", "repair"):
            r = FATE[(_l, _d)][_g]
            n = r["primary"]
            if not n:
                continue
            fin = r["final"]
            print(f"  {_g:<7} {n:>9,} "
                  f"{_pct(n - r['absent_from_graph'], n):>9} "
                  f"{_pct(r['in_pair'], n):>8} "
                  f"{_pct(r['past_floor'], n):>8} "
                  f"{_pct(r['ilp_linked'], n):>8} "
                  f"{_pct(r['gap_gain'], n):>7} "
                  f"{_pct(r['short_kill'], n):>7} "
                  f"{_pct(r['div_gain'], n):>6} "
                  f"{_pct(fin, n):>8} "
                  f"{_pct(r['final_isolated'], n):>9} "
                  f"{r['final_deg_sum'] / fin if fin else 0:>6.2f}",
                  flush=True)
print("\n  in_graph < 100% would mean the ILP solver PRUNES unselected "
      "nodes, in which case 'ISOLATED' cannot be nonzero and the "
      "isolated-node reading is unavailable -- check that column before "
      "reading anything else.", flush=True)

print("\n=== FINAL-OUTPUT DEGREE HISTOGRAM (3 means 3 or more) ===",
      flush=True)
for _d in ("6bba", "44b6", "zh"):
    for _l, _det, _rk in RUN_ARMS:
        if (_l, _d) not in DEGHIST:
            continue
        for _g in ("base", "merged", "repair"):
            h = DEGHIST[(_l, _d)][_g]
            tot = sum(h.values())
            if not tot:
                continue
            print(f"  {_l:<4} {_d:<5} {_g:<7} n={tot:>8,}  "
                  + "  ".join(f"deg{k}={_pct(h.get(k, 0), tot)}"
                              for k in (0, 1, 2, 3)), flush=True)

print("\n=== DID THE ADDED POINTS STEAL LINKS FROM THE BASE POINTS? ===",
      flush=True)
print("  The base points are the SAME points in both arms and carry the "
      "same (t, i) indices, so their ILP-linked sets compare directly. "
      "'lost' = linked under F4 and NOT under M15. This is the "
      "capacity-2 mechanism that would explain 6bba's edge_J falling "
      "0.0037 while its node count rose.", flush=True)
print(f"  {'dataset':<20} {'F4 linked':>12} {'M15':>12} {'lost':>8} "
      f"{'gained':>8} {'net':>9}", flush=True)
_agg = defaultdict(lambda: [0, 0, 0, 0])
for _n in sorted({k[0] for k in BASE_LINKED}):
    if (_n, "F4") not in BASE_LINKED or (_n, "M15") not in BASE_LINKED:
        continue
    _a, _b = BASE_LINKED[(_n, "F4")], BASE_LINKED[(_n, "M15")]
    _lost, _gain = len(_a - _b), len(_b - _a)
    _dm = _n.split("_")[0] if _n.startswith(("6bba", "44b6")) else "zh"
    _ag = _agg[_dm]
    _ag[0] += len(_a)
    _ag[1] += len(_b)
    _ag[2] += _lost
    _ag[3] += _gain
    print(f"  {_n:<20} {len(_a):>12,} {len(_b):>12,} {_lost:>8,} "
          f"{_gain:>8,} {len(_b) - len(_a):>+9,}", flush=True)
print("  " + "-" * 76, flush=True)
for _dm, _ag in sorted(_agg.items()):
    print(f"  {_dm:<20} {_ag[0]:>12,} {_ag[1]:>12,} {_ag[2]:>8,} "
          f"{_ag[3]:>8,} {_ag[1] - _ag[0]:>+9,}   lost "
          f"{_pct(_ag[2], _ag[0])} of F4's links", flush=True)

print("\n=== how to read this ===", flush=True)
print("  The decisive column is ISOLATED: merged nodes that reach the "
      "geff with ZERO edges are pure node-count cost -- they pay the "
      "penalty and buy no edge_J. If that number is large, M15 failed "
      "because the LINKER could not use the points, which is fixable "
      "without a spurious-detection label. If it is small and the "
      "steal table shows base links lost instead, M15 failed because "
      "the added points DISPLACED real links under capacity-2, which "
      "is a different problem and argues the points are wrong after "
      "all. Both readings are consistent with 0.917; this run "
      "separates them.", flush=True)

print("\n=== reading it (stale mergescore verdict text follows) ===",
      flush=True)
print("  The deciding vote is 44b6: competition-style annotation, "
      "our biggest node deficit (-14.8%%), and ep05-clean. zh's dense "
      "GT subsidises added points (the user's observation), so a "
      "zh-only win repeats dcmerge and settles nothing. CLEAN DOMAINS "
      "POSITIVE with a modest node growth = LB candidate on top of "
      "0.925; clean-domains-negative = the point-add line closes with "
      "ep05 too, and what it settles is that the node tax beats the "
      "conversion rate even where the deficit is largest.", flush=True)
print(f"\nMERGESCORE DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)
